import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import warnings
import joblib
import holidays
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuración
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Resumen General", layout="wide")

# --- FUNCIONES DE CARGA DE DATOS ---
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
    except: return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        return None

# --- COMPONENTE VISUAL: FICHA TÉCNICA (SIDEBAR) 
def render_ficha_tecnica(model_data, key, last_train_date):
    """Renderiza la ficha técnica del modelo en el sidebar."""
    if not model_data:
        st.sidebar.info("No hay modelo para esta selección.")
        return

    # CORRECCIÓN AQUÍ: Accedemos al sub-diccionario 'metrics'
    metrics = model_data.get('metrics', {})
    model_obj = model_data.get('model', None)
    
    st.sidebar.divider()
    st.sidebar.header("Ficha Técnica Modelo")
    
    # 1. Info General
    st.sidebar.caption(f"ID: {key}")
    if isinstance(last_train_date, datetime):
        st.sidebar.text(f"Entrenado: {last_train_date.strftime('%d/%m/%Y')}")
    
    # 2. Hiperparámetros (Top 4)
    if model_obj:
        params = model_obj.get_params()
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Árboles", params.get('n_estimators'))
        c1.metric("Profundidad", params.get('max_depth'))
        c2.metric("L. Rate", params.get('learning_rate'))
        c2.metric("Subsample", params.get('subsample'))

    # 3. Métricas Visuales (Radar)
    st.sidebar.subheader("Rendimiento")
    
    radar_vals = [
        metrics.get('accuracy', 0),
        metrics.get('f1_score', 0),
        metrics.get('sensitivity', 0),
        metrics.get('precision', 0)
    ]
    radar_cats = ['Acc', 'F1', 'Recall', 'Prec']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_cats + [radar_cats[0]],
        fill='toself', name='Modelo',
        line_color='#00CC96'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, margin=dict(l=20, r=20, t=10, b=10),
        height=200, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10)
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)
    
    # Detalle Numérico
    st.sidebar.caption("Métricas Clave")
    # CORRECCIÓN DEL ERROR: Accedemos a rmse dentro de metrics
    rmse_val = metrics.get('rmse', 0)
    st.sidebar.metric("RMSE (Error Medio)", f"{rmse_val:.2f}")
    st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0):.2f}")


# --- LÓGICA DE PREDICCIÓN (RECURSIVA) 
def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    # Generación de las 16 variables exactas que espera el modelo
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    mes = fecha_obj.month; dia_sem = fecha_obj.weekday()
    
    row['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    row['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    
    es_holidays = holidays.Spain(years=[fecha_obj.year])
    row['es_festivo'] = int(fecha_obj in es_holidays)
    row['temp_gripe'] = int(mes in [10, 11, 12, 1, 2])
    row['temp_alergia'] = int(mes in [3, 4, 5, 6])
    
    t_media = 15.0
    if df_clima is not None:
        match = df_clima[df_clima['Fecha'] == fecha_obj]
        if not match.empty: t_media = match.iloc[0]['Temperatura_Media']
    row['Temperatura_Media'] = t_media
    
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
    
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_recursivo(model, df_hist_prod, df_clima, dias_futuros):
    ult_fecha = df_hist_prod['Fecha'].max()
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

# --- APP PRINCIPAL ---
st.title("Resumen General y Pronóstico de Demanda")
df_total = cargar_datos(); datos_modelos = cargar_modelos(); df_clima = cargar_clima()

if df_total is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Configuración")
    
    # 1. Filtros
    farm_sel = st.sidebar.selectbox("Farmacia:", ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique())))
    
    if farm_sel == 'Todas':
        prods_disp = sorted(df_total['Producto'].unique())
    else:
        prods_disp = sorted(df_total[df_total['Farmacia_ID'] == farm_sel]['Producto'].unique())
    
    prod_sel = st.sidebar.selectbox("Producto de Interés:", prods_disp)

    # 2. Render Ficha Técnica (Sidebar)
    if datos_modelos:
        farm_ref = farm_sel if farm_sel != 'Todas' else df_total[df_total['Producto']==prod_sel]['Farmacia_ID'].iloc[0]
        key_mod = f"{farm_ref}::{prod_sel}"
        info_mod = datos_modelos['modelos'].get(key_mod)
        fecha_ent = datos_modelos.get('fecha_entrenamiento')
        render_ficha_tecnica(info_mod, key_mod, fecha_ent)
    else:
        st.sidebar.warning("Modelos no cargados.")

    # --- CUERPO ---
    f_min, f_max = df_total['Fecha'].min(), df_total['Fecha'].max()
    with st.expander("Filtros Temporales (Histórico)"):
        rango = st.date_input("Rango Fechas:", [f_min, f_max])
    
    df_fil = df_total.copy()
    if len(rango)==2: df_fil = df_fil[(df_fil['Fecha']>=rango[0]) & (df_fil['Fecha']<=rango[1])]
    if farm_sel != 'Todas': df_fil = df_fil[df_fil['Farmacia_ID'] == farm_sel]

    tab1, tab2, tab3 = st.tabs(["KPIs Generales", "Análisis Estacional", "Motor de IA (Pronóstico)"])
    
    with tab1:
        st.markdown(f"### Vista General: {farm_sel}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ventas Totales", f"{df_fil['Total_Venta_€'].sum():,.0f} €")
        c2.metric("Unidades", f"{df_fil['Cantidad'].sum():,.0f}")
        c3.metric("Ticket Promedio", f"{df_fil['Total_Venta_€'].mean():,.2f} €")
        
        st.altair_chart(alt.Chart(df_fil).mark_bar().encode(
            x='Categoria', y='sum(Total_Venta_€)', color='Categoria'
        ).interactive(), use_container_width=True)

    with tab2:
        st.subheader(f"Análisis Histórico: {prod_sel}")
        df_p = df_total[(df_total['Producto'] == prod_sel) & (df_total['Farmacia_ID'] == (farm_sel if farm_sel != 'Todas' else df_total[df_total['Producto']==prod_sel]['Farmacia_ID'].iloc[0]))]
        
        if not df_p.empty:
            st.altair_chart(alt.Chart(df_p).mark_line().encode(x='Fecha', y='Cantidad').interactive(), use_container_width=True)
            
            df_decomp = df_p.groupby('Fecha')['Cantidad'].sum().asfreq('D').fillna(0)
            if len(df_decomp) > 365*2:
                res = seasonal_decompose(df_decomp, model='additive', period=365)
                c_d1, c_d2 = st.columns(2)
                c_d1.line_chart(res.trend, height=200); c_d1.caption("Tendencia")
                c_d2.line_chart(res.seasonal.iloc[:365], height=200); c_d2.caption("Estacionalidad")
            else:
                st.info("Datos insuficientes para descomposición.")

    with tab3:
        st.subheader(f"Predicción IA: {prod_sel}")
        dias = st.slider("Horizonte (días):", 7, 90, 30)
        
        if st.button("Generar Pronóstico", type="primary"):
            if datos_modelos and info_mod:
                with st.spinner("Calculando..."):
                    df_hist = df_total[(df_total['Producto']==prod_sel) & (df_total['Farmacia_ID']==farm_ref)]
                    df_fut = predecir_recursivo(info_mod['model'], df_hist, df_clima, dias)
                    
                    st.success("Cálculo completado")
                    
                    # Gráfica
                    df_hist_plot = df_hist[['Fecha', 'Cantidad']].rename(columns={'Fecha':'ds','Cantidad':'y'})
                    df_hist_plot['Tipo'] = 'Histórico'
                    df_hist_plot['ds'] = pd.to_datetime(df_hist_plot['ds'])
                    
                    df_fut['y'] = df_fut['Prediccion']; df_fut['Tipo'] = 'Predicción IA'
                    df_fut['ds'] = pd.to_datetime(df_fut['ds'])
                    
                    min_date = df_fut['ds'].min() - timedelta(days=90)
                    df_plot = pd.concat([df_hist_plot[df_hist_plot['ds']>min_date], df_fut])
                    
                    c = alt.Chart(df_plot).mark_line().encode(
                        x='ds', y='y', color='Tipo', strokeDash='Tipo'
                    ).interactive()
                    st.altair_chart(c, use_container_width=True)
                    
                    # Feature Importance
                    if 'importance' in info_mod:
                        # Traducción simple
                        traducciones = {
                            'lag_1': 'Ventas Ayer', 'roll_mean_7': 'Tendencia 7d', 
                            'Temperatura_Media': 'Temperatura', 'es_festivo': 'Festivo',
                            'temp_gripe': 'Temporada Gripe', 'mes_sin': 'Ciclo Anual'
                        }
                        df_imp = info_mod['importance'].head(5).copy()
                        df_imp['Factor'] = df_imp['Impulsor'].map(traducciones).fillna(df_imp['Impulsor'])
                        
                        st.caption("Factores de influencia")
                        st.altair_chart(alt.Chart(df_imp).mark_bar(color='#00CC96').encode(
                            x='Importancia', y=alt.Y('Factor', sort='-x')
                        ), use_container_width=True)
            else:
                st.error("No hay modelo disponible para esta combinación.")