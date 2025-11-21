import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
import warnings
import joblib
import holidays
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuración
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Resumen General", layout="wide")

# --- FUNCIONES DE DATOS ---
@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encuentra {file_name}")
        return None

@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except:
        return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        datos_modelos = joblib.load(file_name)
        st.sidebar.success(f"Modelos Premium cargados (Fecha: {datos_modelos['fecha_entrenamiento'].strftime('%d-%m-%Y')})")
        return datos_modelos
    except FileNotFoundError:
        st.error(f"Error: No se encuentra {file_name}")
        return None

# --- LÓGICA DE PREDICCIÓN PREMIUM (RECURSIVA) ---
# Esta función replica exactamente la ingeniería de features del entrenamiento

def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    """Genera las 16 features premium para un solo día futuro."""
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    
    # 1. Ciclos Temporales
    mes = fecha_obj.month
    dia_sem = fecha_obj.weekday()
    row['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    row['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    
    # 2. Festivos y Temporadas
    es_holidays = holidays.Spain(years=[fecha_obj.year])
    row['es_festivo'] = int(fecha_obj in es_holidays)
    row['temp_gripe'] = int(mes in [10, 11, 12, 1, 2])
    row['temp_alergia'] = int(mes in [3, 4, 5, 6])
    
    # 3. Clima
    t_media = 15.0
    if df_clima is not None:
        match = df_clima[df_clima['Fecha'] == fecha_obj]
        if not match.empty: t_media = match.iloc[0]['Temperatura_Media']
    row['Temperatura_Media'] = t_media
    
    # 4. Lags y Rolling (Usando el historial acumulado)
    vals = historia_y
    row['lag_1'] = vals[-1] if len(vals) >= 1 else 0
    row['lag_2'] = vals[-2] if len(vals) >= 2 else 0
    row['lag_7'] = vals[-7] if len(vals) >= 7 else 0
    row['lag_14'] = vals[-14] if len(vals) >= 14 else 0
    
    row['roll_mean_7'] = pd.Series(vals).rolling(7).mean().iloc[-1] if len(vals)>=7 else 0
    row['roll_mean_28'] = pd.Series(vals).rolling(28).mean().iloc[-1] if len(vals)>=28 else 0
    row['roll_std_7'] = pd.Series(vals).rolling(7).std().iloc[-1] if len(vals)>=7 else 0
    
    # Tendencia
    rm7 = row['roll_mean_7'].values[0]
    rm7_prev = pd.Series(vals).rolling(7).mean().iloc[-8] if len(vals)>=8 else rm7
    row['tendencia_semanal'] = rm7 - rm7_prev
    
    # Orden exacto de columnas (16 features)
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_recursivo(model, df_hist_prod, df_clima, dias_futuros, fecha_max):
    """Bucle que predice día a día."""
    historia = list(df_hist_prod.sort_values('Fecha')['Cantidad'].values)
    predicciones = []
    fechas = []
    
    for i in range(dias_futuros):
        fecha_futura = fecha_max + timedelta(days=i+1)
        
        # Crear features
        X_test = crear_features_un_paso(historia, fecha_futura, df_clima)
        
        # Predecir
        y_pred = max(0, model.predict(X_test)[0])
        
        predicciones.append(int(round(y_pred)))
        fechas.append(fecha_futura)
        historia.append(y_pred) # Actualizar historia
        
    return pd.DataFrame({'ds': fechas, 'Prediccion': predicciones})

# --- APP ---
st.title("Resumen General y Pronóstico de Demanda")

df_total = cargar_datos()
datos_modelos = cargar_modelos()
df_clima = cargar_clima()

if df_total is not None:
    # Sidebar
    st.sidebar.title("Menú de Filtros")
    fecha_min, fecha_max = df_total['Fecha'].min(), df_total['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Rango de fechas:", [fecha_min, fecha_max])
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Farmacia:", lista_farmacias)
    
    df_filtered = df_total.copy()
    if len(rango_fechas) == 2:
        df_filtered = df_filtered[(df_filtered['Fecha'] >= rango_fechas[0]) & (df_filtered['Fecha'] <= rango_fechas[1])]
    if farmacia_sel != 'Todas':
        df_filtered = df_filtered[df_filtered['Farmacia_ID'] == farmacia_sel]

    # Pestañas
    tab1, tab2, tab3 = st.tabs(["KPIs y Métricas", "Análisis de Épocas", "Pronóstico de IA"])
    
    with tab1:
        st.header(f"Métricas para: {farmacia_sel}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas Totales", f"{df_filtered['Total_Venta_€'].sum():,.2f} €")
        col2.metric("Unidades", f"{df_filtered['Cantidad'].sum():,.0f}")
        col3.metric("Transacciones", f"{len(df_filtered):,.0f}")
        st.divider()
        st.subheader("Ventas por Categoría")
        chart = alt.Chart(df_filtered).mark_bar().encode(
            x='Categoria', y='sum(Total_Venta_€)', color='Categoria', tooltip=['Categoria', 'sum(Total_Venta_€)']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.header("Análisis Estacional")
        chart_line = alt.Chart(df_filtered).mark_line().encode(
            x='yearmonth(Fecha)', y='sum(Cantidad)', color='Categoria'
        ).interactive()
        st.altair_chart(chart_line, use_container_width=True)
        
        st.divider()
        st.subheader("Descomposición de Serie Temporal")
        prods = sorted(df_filtered['Producto'].unique())
        if prods:
            p_sel = st.selectbox("Producto:", prods)
            df_p = df_filtered[df_filtered['Producto'] == p_sel].groupby('Fecha')['Cantidad'].sum()
            df_p.index = pd.to_datetime(df_p.index)
            df_p = df_p.asfreq('D').fillna(0)
            
            if len(df_p) > 365*2:
                res = seasonal_decompose(df_p, model='additive', period=365)
                st.line_chart(res.trend, height=200)
                st.caption("Tendencia")
                st.line_chart(res.seasonal.iloc[:365], height=200)
                st.caption("Estacionalidad (1 año)")

    with tab3:
        st.header("Pronóstico Avanzado")
        col_p, col_d = st.columns(2)
        prod_pred = col_p.selectbox("Producto:", sorted(df_total['Producto'].unique()))
        days_pred = col_d.slider("Días a predecir:", 7, 90, 30)
        
        if st.button("Generar Pronóstico", type="primary"):
            if datos_modelos:
                # Lógica para 'Todas' las farmacias (usamos una como referencia para la demo)
                farm_ref = farmacia_sel
                if farmacia_sel == 'Todas':
                    farm_ref = df_total[df_total['Producto'] == prod_pred]['Farmacia_ID'].iloc[0]
                    st.info(f"Nota: Mostrando pronóstico para {farm_ref} como referencia.")

                key = f"{farm_ref}::{prod_pred}"
                model_info = datos_modelos['modelos'].get(key)
                
                if model_info:
                    with st.spinner("Calculando predicción día a día..."):
                        # Historial específico
                        df_hist = df_total[
                            (df_total['Producto'] == prod_pred) & 
                            (df_total['Farmacia_ID'] == farm_ref)
                        ]
                        
                        # Predicción Recursiva
                        df_futuro = predecir_recursivo(
                            model_info['model'], df_hist, df_clima, days_pred, df_total['Fecha'].max()
                        )
                        
                        # Visualización
                        st.success("Pronóstico Completado")
                        
                        c1, c2 = st.columns([1, 3])
                        c1.metric("RMSE (Error)", f"{model_info['rmse']:.2f}", help="Error promedio en unidades.")
                        
                        if 'importance' in model_info:
                            c2.altair_chart(alt.Chart(model_info['importance'].head(10)).mark_bar().encode(
                                x='Importancia', y=alt.Y('Impulsor', sort='-x')
                            ).properties(title="Factores Clave"), use_container_width=True)
                        
                        st.subheader("Proyección")
                        df_hist_plot = df_hist[['Fecha', 'Cantidad']].rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
                        df_hist_plot['Tipo'] = 'Real'
                        df_futuro_plot = df_futuro.rename(columns={'Prediccion': 'y'})
                        df_futuro_plot['Tipo'] = 'Predicción'
                        
                        start_plot = df_futuro_plot['ds'].min() - timedelta(days=365)
                        df_plot = pd.concat([df_hist_plot[pd.to_datetime(df_hist_plot['ds']) > start_plot], df_futuro_plot])
                        
                        chart_pred = alt.Chart(df_plot).mark_line().encode(
                            x='ds', y='y', color='Tipo', strokeDash='Tipo'
                        ).interactive()
                        st.altair_chart(chart_pred, use_container_width=True)
                        
                        st.dataframe(df_futuro.set_index('ds'), use_container_width=True)
                else:
                    st.warning("No hay modelo entrenado para esta combinación.")
            else:
                st.error("Modelos no cargados.")