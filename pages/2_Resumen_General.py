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
        st.error(f"¡ERROR! No se encontró '{file_name}'.")
        return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        datos_modelos = joblib.load(file_name)
        st.sidebar.success(f"Modelos Premium cargados (Entrenados el {datos_modelos['fecha_entrenamiento'].strftime('%d-%m-%Y')})")
        return datos_modelos
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró '{file_name}'.")
        return None

@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    try:
        df_clima = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha']).dt.date
        return df_clima
    except FileNotFoundError:
        return None

# --- INGENIERÍA DE CARACTERÍSTICAS (Debe coincidir con train_models.py) ---

def crear_features_un_paso(df_window, fecha_objetivo, df_clima):
    """
    Calcula las features para UNA sola fila (la fecha objetivo) basándose
    en la ventana de datos históricos proporcionada.
    """
    # 1. Crear fila base
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_objetivo)]})
    
    # 2. Features Temporales
    row['mes'] = row['ds'].dt.month
    row['dia_semana'] = row['ds'].dt.dayofweek
    row['dia_ano'] = row['ds'].dt.dayofyear
    
    # 3. Cíclicas
    row['mes_sin'] = np.sin(2 * np.pi * row['mes'] / 12)
    row['mes_cos'] = np.cos(2 * np.pi * row['mes'] / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * row['dia_semana'] / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * row['dia_semana'] / 7)
    
    # 4. Festivos
    es_holidays = holidays.Spain(years=[fecha_objetivo.year])
    row['es_festivo'] = row['ds'].isin(es_holidays).astype(int)
    
    # 5. Temporadas
    row['temp_gripe'] = row['mes'].isin([10, 11, 12, 1, 2]).astype(int)
    row['temp_alergia'] = row['mes'].isin([3, 4, 5, 6]).astype(int)
    
    # 6. Clima (Lookup)
    t_media = 15.0
    if df_clima is not None:
        match = df_clima[df_clima['Fecha'] == fecha_objetivo]
        if not match.empty:
            t_media = match.iloc[0]['Temperatura_Media']
    row['Temperatura_Media'] = t_media
    
    # --- FEATURES COMPLEJAS (Lags y Rolling) ---
    # Necesitamos los datos históricos (df_window) para calcular esto
    # df_window debe estar ordenado y terminar en el día anterior a fecha_objetivo
    
    historia_y = df_window['y'].values
    
    # Lags
    row['lag_1'] = historia_y[-1] if len(historia_y) >= 1 else 0
    row['lag_2'] = historia_y[-2] if len(historia_y) >= 2 else 0
    row['lag_7'] = historia_y[-7] if len(historia_y) >= 7 else 0
    row['lag_14'] = historia_y[-14] if len(historia_y) >= 14 else 0
    
    # Rolling Means
    row['roll_mean_7'] = pd.Series(historia_y).rolling(window=7).mean().iloc[-1] if len(historia_y) >= 7 else 0
    row['roll_mean_28'] = pd.Series(historia_y).rolling(window=28).mean().iloc[-1] if len(historia_y) >= 28 else 0
    
    # Rolling Std (Volatilidad)
    row['roll_std_7'] = pd.Series(historia_y).rolling(window=7).std().iloc[-1] if len(historia_y) >= 7 else 0
    
    # Tendencia
    rm7 = row['roll_mean_7'].values[0]
    rm7_prev = pd.Series(historia_y).rolling(window=7).mean().iloc[-8] if len(historia_y) >= 8 else rm7
    row['tendencia_semanal'] = rm7 - rm7_prev
    
    return row

def predecir_recursivo(model, df_historico, df_clima, dias_a_pronosticar):
    """
    Realiza predicciones día a día, actualizando el historial con la propia predicción
    para calcular correctamente los lags futuros.
    """
    # Preparamos historial inicial
    historial = df_historico.copy().sort_values('Fecha')
    historial = historial.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    
    ult_fecha = pd.to_datetime(historial['ds'].max())
    predicciones = []
    fechas_futuras = []
    
    feature_order = [
        'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
        'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
        'lag_1', 'lag_2', 'lag_7', 'lag_14',
        'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal'
    ]
    
    # Bucle día a día
    for i in range(dias_a_pronosticar):
        fecha_obj = ult_fecha + timedelta(days=i+1)
        
        # Crear features para ESTE día usando el historial acumulado
        row_features = crear_features_un_paso(historial, fecha_obj, df_clima)
        
        # Predecir
        X = row_features[feature_order]
        y_pred = model.predict(X)[0]
        y_pred = max(0, y_pred) # No ventas negativas
        
        # Guardar
        predicciones.append(int(round(y_pred)))
        fechas_futuras.append(fecha_obj)
        
        # AÑADIR PREDICCIÓN AL HISTORIAL (para que el siguiente loop la use como lag)
        nueva_fila = pd.DataFrame({'ds': [fecha_obj], 'y': [y_pred]})
        historial = pd.concat([historial, nueva_fila], ignore_index=True)
        
    return pd.DataFrame({'ds': fechas_futuras, 'Prediccion': predicciones})

# --- APP ---

st.title("Resumen General y Pronóstico de Demanda")

df_total = cargar_datos()
datos_modelos = cargar_modelos()
df_clima = cargar_clima()

if df_total is not None:
    # Sidebar Filtros
    st.sidebar.title("Menú de Filtros")
    fecha_min, fecha_max = df_total['Fecha'].min(), df_total['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Rango de fechas:", [fecha_min, fecha_max])
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Farmacia:", lista_farmacias)
    
    # Filtrado
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
        # Gráfico de líneas simple para ver tendencias
        chart_line = alt.Chart(df_filtered).mark_line().encode(
            x='yearmonth(Fecha)', y='sum(Cantidad)', color='Categoria'
        ).interactive()
        st.altair_chart(chart_line, use_container_width=True)
        
        # Descomposición (Simplified for robustness)
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
                st.caption("Tendencia a largo plazo")
                st.line_chart(res.seasonal.iloc[:365], height=200) # Solo un año para ver el patrón
                st.caption("Patrón Estacional (Zoom 1 año)")
            else:
                st.warning("Datos insuficientes para descomposición.")

    with tab3:
        st.header("Pronóstico Avanzado (Recursivo)")
        
        col_p, col_d = st.columns(2)
        prod_pred = col_p.selectbox("Producto:", sorted(df_total['Producto'].unique()))
        days_pred = col_d.slider("Días a predecir:", 7, 90, 30)
        
        if st.button("Generar Pronóstico", type="primary"):
            if datos_modelos:
                key = f"{farmacia_sel}::{prod_pred}"
                # Si es 'Todas', cogemos la primera farmacia que tenga el producto para la demo
                # (En una app real, habría que agregar o entrenar un modelo global)
                if farmacia_sel == 'Todas':
                    # Buscar una farmacia válida para este producto
                    valid_farm = df_total[df_total['Producto'] == prod_pred]['Farmacia_ID'].iloc[0]
                    key = f"{valid_farm}::{prod_pred}"
                    st.info(f"Nota: Pronosticando para {valid_farm} como referencia.")
                
                model_info = datos_modelos['modelos'].get(key)
                
                if model_info:
                    with st.spinner("Calculando predicción día a día..."):
                        # Obtener historial completo para las features
                        df_hist = df_total[
                            (df_total['Producto'] == prod_pred) & 
                            (df_total['Farmacia_ID'] == (farmacia_sel if farmacia_sel != 'Todas' else valid_farm))
                        ]
                        
                        df_futuro = predecir_recursivo(
                            model_info['model'], 
                            df_hist, 
                            df_clima, 
                            days_pred
                        )
                        
                        # Visualización
                        st.success("¡Pronóstico Completado!")
                        
                        # Métricas
                        c1, c2 = st.columns([1, 3])
                        c1.metric("RMSE (Error)", f"{model_info['rmse']:.2f}", help="Error promedio del modelo en unidades.")
                        
                        # Gráfico Importancia
                        c2.altair_chart(alt.Chart(model_info['importance'].head(10)).mark_bar().encode(
                            x='Importancia', y=alt.Y('Impulsor', sort='-x'), tooltip=['Impulsor', 'Importancia']
                        ).properties(title="Factores Clave del Modelo"), use_container_width=True)
                        
                        # Gráfico Resultado
                        st.subheader("Proyección de Ventas")
                        
                        # Unir real + predicción para gráfico continuo
                        df_hist_plot = df_hist[['Fecha', 'Cantidad']].rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
                        df_hist_plot['Tipo'] = 'Real'
                        df_futuro_plot = df_futuro.rename(columns={'Prediccion': 'y'})
                        df_futuro_plot['Tipo'] = 'Predicción'
                        
                        # Mostrar solo último año + futuro
                        start_plot = df_futuro_plot['ds'].min() - timedelta(days=365)
                        df_plot = pd.concat([df_hist_plot[pd.to_datetime(df_hist_plot['ds']) > start_plot], df_futuro_plot])
                        
                        chart_pred = alt.Chart(df_plot).mark_line().encode(
                            x='ds', y='y', color='Tipo', strokeDash='Tipo'
                        ).interactive()
                        st.altair_chart(chart_pred, use_container_width=True)
                        
                else:
                    st.error("No hay modelo entrenado para esta combinación.")
            else:
                st.error("Modelos no cargados.")