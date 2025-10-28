import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
import warnings
import joblib # <-- ¡Nueva importación!
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Alerta de Stock (IA)", page_icon="🚨", layout="wide")

# --- FUNCIONES DE DATOS (CON CACHÉ) ---

@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    """Carga los datos base desde el CSV."""
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        df['Fecha'] = df['Fecha'].dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

@st.cache_data
def simular_stock_actual(_df_total):
    """
    Simula un inventario de "Stock Actual" para cada producto/farmacia.
    """
    st.info("Simulando inventario de stock actual... (se ejecuta una vez)", icon="📦")
    
    df_ventas_diarias = _df_total.groupby(['Farmacia_ID', 'Producto'])['Cantidad'].mean().reset_index()
    df_ventas_diarias = df_ventas_diarias.rename(columns={'Cantidad': 'Venta_Media_Diaria'})
    
    np.random.seed(123)
    dias_stock_simulados = np.random.randint(1, 21, size=len(df_ventas_diarias))
    
    df_ventas_diarias['Stock_Actual'] = (df_ventas_diarias['Venta_Media_Diaria'] * dias_stock_simulados).round(0).astype(int)
    
    return df_ventas_diarias[['Farmacia_ID', 'Producto', 'Stock_Actual']]

# --- FUNCIONES DEL MODELO DE IA (AHORA OPTIMIZADAS) ---

@st.cache_resource # <-- Cache de RECURSO (para el archivo del modelo)
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    """Carga los modelos pre-entrenados desde el archivo joblib."""
    try:
        datos_modelos = joblib.load(file_name)
        st.success(f"Modelos de IA cargados (entrenados el {datos_modelos['fecha_entrenamiento'].strftime('%Y-%m-%d %H:%M')})")
        return datos_modelos
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        st.info("Por favor, ejecuta el script 'train_models.py' en tu terminal para generar el archivo de modelos.")
        return None

def crear_caracteristicas_temporales(df_in):
    """Crea características de tiempo para el modelo de ML."""
    df_out = df_in.copy()
    df_out['ds'] = pd.to_datetime(df_out['ds']) 
    df_out['mes'] = df_out['ds'].dt.month
    df_out['dia_del_ano'] = df_out['ds'].dt.dayofyear
    df_out['semana_del_ano'] = df_out['ds'].dt.isocalendar().week.astype(int)
    df_out['dia_de_la_semana'] = df_out['ds'].dt.dayofweek
    df_out['ano'] = df_out['ds'].dt.year
    df_out['trimestre'] = df_out['ds'].dt.quarter
    return df_out

def generar_pronostico(model, dias_a_pronosticar, fecha_maxima_historica):
    """Genera la predicción de demanda usando un modelo YA ENTRENADO."""
    fecha_max_dt = pd.to_datetime(fecha_maxima_historica)
    fechas_futuras = pd.date_range(
        start=fecha_max_dt + pd.Timedelta(days=1),
        periods=dias_a_pronosticar, freq='D'
    )
    df_futuro = pd.DataFrame({'ds': fechas_futuras})
    df_futuro_preparado = crear_caracteristicas_temporales(df_futuro)
    
    features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
    prediccion_futura = model.predict(df_futuro_preparado[features])
    prediccion_futura[prediccion_futura < 0] = 0
    
    return prediccion_futura.sum().round(0).astype(int)

# --- INTERFAZ DE STREAMLIT ---

st.title("🚨 Sistema de Alerta de Stock (IA)")
st.markdown("""
Esta es la herramienta principal. Compara el **Stock Actual** (simulado) con la 
**Demanda Futura** (pronosticada por la IA) para identificar riesgos de rotura de stock.
""")

df_total = cargar_datos()
datos_modelos = cargar_modelos() # Carga los modelos al inicio

if df_total is not None and datos_modelos is not None:
    
    df_stock_actual = simular_stock_actual(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Parámetros de Alerta")
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox(
        "Selecciona Farmacia:",
        options=lista_farmacias,
        key='alerta_farmacia'
    )
    
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    try:
        default_index = lista_categorias.index('Antigripal')
    except ValueError:
        default_index = 0 
    cat_sel = st.sidebar.selectbox(
        "Selecciona Categoría:",
        options=lista_categorias,
        index=default_index,
        key='alerta_categoria'
    )
    
    st.sidebar.divider()
    
    dias_a_pronosticar = st.sidebar.slider(
        "Días a Pronosticar (Horizonte):",
        min_value=1, max_value=30, value=7,
        help="¿Para cuántos días quieres predecir la demanda? (Ej. 7 días = demanda semanal)"
    )
    stock_seguridad_pct = st.sidebar.slider(
        "Stock de Seguridad (%):",
        min_value=0, max_value=100, value=20,
        help="Un colchón adicional sobre la demanda pronosticada (Ej. 20%)"
    )

    # --- LÓGICA DE EJECUCIÓN ---
    if st.button("Generar Reporte de Alertas", type="primary", use_container_width=True):
        
        # Filtrar la lista de productos a revisar
        df_a_revisar = df_stock_actual.copy()
        
        if farmacia_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Farmacia_ID'] == farmacia_sel]
            
        df_a_revisar = df_a_revisar.merge(
            df_total[['Producto', 'Categoria']].drop_duplicates(),
            on='Producto', how='left'
        )
        
        if cat_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Categoria'] == cat_sel]

        if df_a_revisar.empty:
            st.warning("No hay productos/farmacias que coincidan con los filtros.")
        else:
            fecha_max_hist = df_total['Fecha'].max()
            
            # 1. Cargar el diccionario de modelos UNA SOLA VEZ
            modelos_cargados = datos_modelos['modelos']
            
            resultados = []
            barra_progreso = st.progress(0, text="Iniciando análisis...")
            
            total_productos = len(df_a_revisar)
            
            # 2. Iterar por cada producto/farmacia a revisar
            for i, row in enumerate(df_a_revisar.itertuples()):
                
                farmacia_id = row.Farmacia_ID
                producto = row.Producto
                stock_actual = row.Stock_Actual
                
                barra_progreso.progress((i+1)/total_productos, text=f"Consultando IA: {producto} en {farmacia_id}...")
                
                # 3. Buscar el modelo (¡sin entrenar!)
                clave_modelo = f"{farmacia_id}::{producto}"
                modelo = modelos_cargados.get(clave_modelo)
                
                if modelo is None:
                    demanda_predicha = 0
                    alerta = "Sin Datos (IA)"
                    stock_necesario = 0
                else:
                    # 4. Generar pronóstico (rápido)
                    demanda_predicha = generar_pronostico(modelo, dias_a_pronosticar, fecha_max_hist)
                    
                    # 5. Aplicar Lógica de Negocio
                    stock_seguridad_unidades = demanda_predicha * (stock_seguridad_pct / 100)
                    stock_necesario = demanda_predicha + stock_seguridad_unidades
                    
                    if stock_actual < stock_necesario:
                        alerta = "ALERTA (Pedir)"
                    else:
                        alerta = "OK"
                
                resultados.append({
                    "Farmacia": farmacia_id,
                    "Producto": producto,
                    "Stock Actual": stock_actual,
                    f"Demanda Predicha ({dias_a_pronosticar} días)": demanda_predicha,
                    "Stock de Seguridad (%)": stock_seguridad_pct,
                    "Stock Necesario": int(stock_necesario),
                    "Estado": alerta
                })
            
            barra_progreso.empty() 
            
            # --- Mostrar Resultados ---
            st.success("¡Análisis de stock completado!")
            
            df_resultados = pd.DataFrame(resultados)
            
            def estilizar_alertas(fila):
                if fila.Estado == "ALERTA (Pedir)":
                    return ['background-color: #FF4B4B; color: white'] * len(fila)
                elif fila.Estado == "Sin Datos (IA)":
                    return ['background-color: #A9A9A9; color: white'] * len(fila)
                else:
                    return [''] * len(fila)

            st.dataframe(
                df_resultados.style.apply(estilizar_alertas, axis=1),
                use_container_width=True
            )
else:
    st.error("Error al cargar los datos. Revisa el archivo CSV y asegúrate de haber ejecutado 'train_models.py'.")