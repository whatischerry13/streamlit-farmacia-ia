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

warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Resumen General", layout="wide")

# --- FUNCIONES DE DATOS ---
@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except FileNotFoundError: return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        datos = joblib.load(file_name)
        st.sidebar.success(f"Modelos Premium activos (v.{datos['fecha_entrenamiento'].strftime('%d%m')})")
        return datos
    except: return None

@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except: return None

# --- LÓGICA DE PREDICCIÓN PREMIUM (RECURSIVA 16 FEATURES) ---
# Esta función crea EXACTAMENTE las columnas que pide el error

def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    
    # 1. Ciclos Temporales (Seno/Coseno)
    mes = fecha_obj.month; dia_sem = fecha_obj.weekday()
    row['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    row['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    
    # 2. Festivos
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
    
    # 4. Lags y Rolling (Complejos)
    vals = historia_y
    row['lag_1'] = vals[-1] if len(vals)>=1 else 0
    row['lag_2'] = vals[-2] if len(vals)>=2 else 0
    row['lag_7'] = vals[-7] if len(vals)>=7 else 0
    row['lag_14'] = vals[-14] if len(vals)>=14 else 0
    
    row['roll_mean_7'] = pd.Series(vals).rolling(7).mean().iloc[-1] if len(vals)>=7 else 0
    row['roll_mean_28'] = pd.Series(vals).rolling(28).mean().iloc[-1] if len(vals)>=28 else 0
    row['roll_std_7'] = pd.Series(vals).rolling(7).std().iloc[-1] if len(vals)>=7 else 0
    
    rm7 = row['roll_mean_7'].values[0]
    rm7_prev = pd.Series(vals).rolling(7).mean().iloc[-8] if len(vals)>=8 else rm7
    row['tendencia_semanal'] = rm7 - rm7_prev
    
    # Orden EXACTO del modelo Premium
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_recursivo(model, df_hist_prod, df_clima, dias_futuros):
    # Última fecha conocida
    ult_fecha = df_hist_prod['Fecha'].max()
    # Historial ordenado
    historia = list(df_hist_prod.sort_values('Fecha')['Cantidad'].values)
    
    predicciones = []
    fechas = []
    
    for i in range(dias_futuros):
        fecha_futura = ult_fecha + timedelta(days=i+1)
        X_test = crear_features_un_paso(historia, fecha_futura, df_clima)
        y_pred = max(0, model.predict(X_test)[0])
        
        predicciones.append(int(round(y_pred)))
        fechas.append(fecha_futura)
        historia.append(y_pred)
        
    return pd.DataFrame({'ds': fechas, 'Prediccion': predicciones})

# --- APP ---
st.title("Resumen General y Pronóstico de Demanda")
df_total = cargar_datos(); datos_modelos = cargar_modelos(); df_clima = cargar_clima()

if df_total is not None:
    # Filtros
    st.sidebar.header("Filtros Globales")
    f_min, f_max = df_total['Fecha'].min(), df_total['Fecha'].max()
    rango = st.sidebar.date_input("Fechas:", [f_min, f_max])
    farm_sel = st.sidebar.selectbox("Farmacia:", ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique())))
    
    df_fil = df_total.copy()
    if len(rango)==2: df_fil = df_fil[(df_fil['Fecha']>=rango[0]) & (df_fil['Fecha']<=rango[1])]
    if farm_sel != 'Todas': df_fil = df_fil[df_fil['Farmacia_ID'] == farm_sel]

    tab1, tab2, tab3 = st.tabs(["KPIs", "Estacionalidad", "Pronóstico IA"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Ventas", f"{df_fil['Total_Venta_€'].sum():,.0f} €")
        c2.metric("Unidades", f"{df_fil['Cantidad'].sum():,.0f}")
        c3.metric("Transacciones", f"{len(df_fil):,.0f}")
        st.altair_chart(alt.Chart(df_fil).mark_bar().encode(x='Categoria', y='sum(Total_Venta_€)', color='Categoria').interactive(), use_container_width=True)

    with tab2:
        st.subheader("Tendencias")
        st.altair_chart(alt.Chart(df_fil).mark_line().encode(x='yearmonth(Fecha)', y='sum(Cantidad)', color='Categoria').interactive(), use_container_width=True)

    with tab3:
        st.header("Motor de IA Premium")
        c_prod, c_dias = st.columns(2)
        prod = c_prod.selectbox("Producto:", sorted(df_total['Producto'].unique()))
        dias = c_dias.slider("Días a predecir:", 7, 90, 30)
        
        if st.button("Generar Pronóstico", type="primary"):
            if datos_modelos:
                farm_ref = farm_sel if farm_sel != 'Todas' else df_total[df_total['Producto']==prod]['Farmacia_ID'].iloc[0]
                key = f"{farm_ref}::{prod}"
                model_info = datos_modelos['modelos'].get(key)
                
                if model_info:
                    df_hist = df_total[(df_total['Producto']==prod) & (df_total['Farmacia_ID']==farm_ref)]
                    df_fut = predecir_recursivo(model_info['model'], df_hist, df_clima, dias)
                    
                    st.success("Pronóstico Generado")
                    
                    # Métricas
                    c1, c2 = st.columns([1, 2])
                    c1.metric("RMSE (Precisión)", f"{model_info['rmse']:.2f}", help="Menor es mejor")
                    
                    # Gráfico Importancia
                    if 'importance' in model_info:
                        c2.altair_chart(alt.Chart(model_info['importance'].head(7)).mark_bar().encode(
                            x='Importancia', y=alt.Y('Impulsor', sort='-x'), tooltip=['Impulsor', 'Importancia']
                        ).properties(title="Factores de Decisión"), use_container_width=True)

                    # Gráfico Línea
                    df_hist_plot = df_hist[['Fecha', 'Cantidad']].rename(columns={'Fecha':'ds','Cantidad':'y'})
                    df_hist_plot['Tipo'] = 'Real'
                    df_fut['y'] = df_fut['Prediccion']; df_fut['Tipo'] = 'Predicción'
                    
                    min_date = df_fut['ds'].min() - timedelta(days=180)
                    df_plot = pd.concat([df_hist_plot[pd.to_datetime(df_hist_plot['ds'])>min_date], df_fut])
                    
                    st.altair_chart(alt.Chart(df_plot).mark_line().encode(
                        x='ds', y='y', color='Tipo', strokeDash='Tipo'
                    ).interactive(), use_container_width=True)
                    
                    st.dataframe(df_fut[['ds', 'Prediccion']].set_index('ds'), use_container_width=True)

                else: st.warning("Sin modelo para esta combinación.")
            else: st.error("Modelos no cargados.")