import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
import holidays
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Alerta de Stock (IA)", layout="wide")

# --- Colores ---
COLOR_ALTA = "#E62222"
COLOR_MEDIA = "#DAA755"
COLOR_BAJA = "#64A7DD"

# --- Funciones de Carga ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

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
        return joblib.load(file_name)
    except:
        st.error(f"Error: No se encuentra {file_name}")
        return None

@st.cache_data
def simular_stock_actual(_df_total):
    st.info("Sincronizando inventario en tiempo real... (Simulado)")
    df_grp = _df_total.groupby(['Farmacia_ID', 'Producto'])['Cantidad'].mean().reset_index()
    np.random.seed(123)
    df_grp['Stock_Actual'] = (df_grp['Cantidad'] * np.random.randint(3, 20, size=len(df_grp))).astype(int)
    return df_grp[['Farmacia_ID', 'Producto', 'Stock_Actual']]

# --- LÓGICA DE PREDICCIÓN PREMIUM (RECURSIVA - 16 FEATURES) ---
# Esta función es la CLAVE para arreglar el error. Genera las mismas 16 columnas que el entrenamiento.

def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    
    # 1. Ciclos Temporales
    mes = fecha_obj.month; dia_sem = fecha_obj.weekday()
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
    # Aseguramos tener datos suficientes, sino 0
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
    
    # Orden exacto de columnas que espera el modelo (16 features)
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_demanda_futura(modelo, df_hist_prod, df_clima, dias_futuros, fecha_max):
    """Bucle que predice día a día alimentándose a sí mismo."""
    # Extraemos la serie de cantidad histórica
    historia = list(df_hist_prod.sort_values('Fecha')['Cantidad'].values)
    predicciones = []
    
    for i in range(dias_futuros):
        fecha_futura = fecha_max + timedelta(days=i+1)
        
        # Crear features para mañana usando la historia
        X_test = crear_features_un_paso(historia, fecha_futura, df_clima)
        
        # Predecir
        y_pred = max(0, modelo.predict(X_test)[0])
        
        predicciones.append(y_pred)
        historia.append(y_pred)
        
    return np.array(predicciones)

# --- INTERFAZ ---
st.title("Sistema de Alerta de Stock (IA Premium)")
st.info("""
**Herramienta de Priorización Inteligente**
Utiliza modelos XGBoost optimizados con variables climáticas y de tendencia para predecir el agotamiento de stock.
Clasifica automáticamente la urgencia de los pedidos basándose en los días restantes de inventario útil.
""", icon="ℹ️")

df_total = cargar_datos()
df_clima = cargar_clima()
datos_modelos = cargar_modelos()

if df_total is not None and datos_modelos is not None:
    df_stock = simular_stock_actual(df_total)
    
    # Sidebar
    st.sidebar.header("Configuración")
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farm_sel = st.sidebar.selectbox("Farmacia:", lista_farmacias, key='alerta_farm')
    
    lista_cats = sorted(list(df_total['Categoria'].unique()))
    try: idx_cat = lista_cats.index('Antigripal')
    except: idx_cat = 0
    cat_sel = st.sidebar.selectbox("Categoría:", lista_cats, index=idx_cat, key='alerta_cat')
    
    st.sidebar.divider()
    dias_horizonte = st.sidebar.slider("Horizonte de Análisis (Días):", 7, 30, 14)
    stock_seguridad_pct = st.sidebar.slider("Stock de Seguridad (%)", 0, 50, 10)

    # Leyenda
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Niveles de Urgencia**")
    st.sidebar.markdown(f"<div style='padding:5px; background-color:{COLOR_ALTA}; color:white; border-radius:5px; margin-bottom:5px; font-size:0.9rem;'>ALTA: Rotura en &le; 3 días</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='padding:5px; background-color:{COLOR_MEDIA}; color:white; border-radius:5px; margin-bottom:5px; font-size:0.9rem;'>MEDIA: Rotura en &le; 7 días</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='padding:5px; background-color:{COLOR_BAJA}; color:white; border-radius:5px; margin-bottom:5px; font-size:0.9rem;'>BAJA: Rotura prevista</div>", unsafe_allow_html=True)

    if st.button("Analizar Riesgos de Stock", type="primary", use_container_width=True):
        # Filtrar
        df_work = df_stock.copy()
        if farm_sel != 'Todas': df_work = df_work[df_work['Farmacia_ID'] == farm_sel]
        
        prods_cat = df_total[['Producto', 'Categoria']].drop_duplicates()
        df_work = df_work.merge(prods_cat, on='Producto')
        df_work = df_work[df_work['Categoria'] == cat_sel]

        if df_work.empty:
            st.warning("No hay productos para analizar con estos filtros.")
        else:
            # Lógica Principal
            fecha_max_hist = df_total['Fecha'].max()
            modelos = datos_modelos['modelos']
            resultados = []
            
            progreso = st.progress(0, text="Iniciando IA...")
            total_items = len(df_work)
            omitted_count = 0

            for i, row in enumerate(df_work.itertuples()):
                progreso.progress((i+1)/total_items, text=f"Analizando: {row.Producto}...")
                
                key = f"{row.Farmacia_ID}::{row.Producto}"
                model_info = modelos.get(key)
                
                if not model_info:
                    omitted_count += 1
                    continue
                
                # Historial específico para recursión
                df_hist_prod = df_total[
                    (df_total['Farmacia_ID'] == row.Farmacia_ID) & 
                    (df_total['Producto'] == row.Producto)
                ]
                
                # --- PREDICCIÓN RECURSIVA (PREMIUM) ---
                preds = predecir_demanda_futura(model_info['model'], df_hist_prod, df_clima, dias_horizonte, fecha_max_hist)
                demanda_total = preds.sum()
                
                # Cálculo de Rotura
                stock_seg = demanda_total * (stock_seguridad_pct/100)
                stock_util = row.Stock_Actual - stock_seg
                
                dias_rotura = "OK"
                stock_temp = stock_util
                for d, venta_dia in enumerate(preds):
                    stock_temp -= venta_dia
                    if stock_temp <= 0:
                        dias_rotura = f"{d + 1} días"
                        break
                
                prioridad = "OK"
                if dias_rotura != "OK":
                    d_num = int(dias_rotura.split()[0])
                    if d_num <= 3: prioridad = "Alta"
                    elif d_num <= 7: prioridad = "Media"
                    else: prioridad = "Baja"
                
                # Añadimos todos a la tabla de detalle
                resultados.append({
                    "Farmacia": row.Farmacia_ID,
                    "Producto": row.Producto,
                    "Prioridad": prioridad,
                    "Días hasta Rotura": dias_rotura,
                    "Stock Actual": row.Stock_Actual,
                    "Stock Útil": int(stock_util),
                    "Demanda Prevista": int(demanda_total),
                    "RMSE Modelo": f"{model_info['rmse']:.1f}"
                })
            
            progreso.empty()
            
            if omitted_count > 0:
                st.info(f"Nota: {omitted_count} productos omitidos por falta de datos históricos suficientes.")

            if not resultados:
                st.warning("No hay resultados para mostrar.")
            else:
                df_res = pd.DataFrame(resultados)
                
                # Tabla Resumen (Solo columnas clave)
                df_view = df_res[["Farmacia", "Producto", "Prioridad", "Días hasta Rotura"]]
                
                def color_row(row):
                    c = ""
                    if row['Prioridad'] == "Alta": c = COLOR_ALTA
                    elif row['Prioridad'] == "Media": c = COLOR_MEDIA
                    elif row['Prioridad'] == "Baja": c = COLOR_BAJA
                    
                    if c:
                        return [f'background-color: {c}; color: white'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.subheader("Resumen de Prioridades")
                st.dataframe(df_view.style.apply(color_row, axis=1), use_container_width=True)
                
                with st.expander("Ver Detalles Completos"):
                    st.dataframe(df_res, use_container_width=True)

                st.download_button("Descargar Alertas (CSV)", convert_df_to_csv(df_res), "alertas.csv", "text/csv")

else:
    st.error("Error cargando datos iniciales.")