import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
from datetime import datetime
import holidays # <-- Importación necesaria

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Alerta de Stock (IA)", layout="wide")

# --- 1. Paleta de Colores Profesional (Suave) ---
COLOR_ALTA = "#9B2B2B"     # Rojo Oscuro (Dark Red)
COLOR_MEDIA = "#B9770E"    # Ámbar Oscuro (Dark Amber)
COLOR_BAJA = "#4682B4"     # Azul Acero (SteelBlue)

# --- 2. Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- Funciones de Datos y Modelos ---
@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    """Carga los datos base desde el CSV."""
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

@st.cache_data
def simular_stock_actual(_df_total):
    """Simula un inventario de 'Stock Actual' para cada producto/farmacia."""
    st.info("Simulando inventario de stock actual... (se ejecuta una vez)")
    df_ventas_diarias = _df_total.groupby(['Farmacia_ID', 'Producto'])['Cantidad'].mean().reset_index()
    df_ventas_diarias = df_ventas_diarias.rename(columns={'Cantidad': 'Venta_Media_Diar'})
    np.random.seed(123)
    dias_stock_simulados = np.random.randint(1, 21, size=len(df_ventas_diarias))
    df_ventas_diarias['Stock_Actual'] = (df_ventas_diarias['Venta_Media_Diar'] * dias_stock_simulados).round(0).astype(int)
    return df_ventas_diarias[['Farmacia_ID', 'Producto', 'Stock_Actual']]

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    """Carga los modelos pre-entrenados desde el archivo joblib."""
    try:
        datos_modelos = joblib.load(file_name)
        return datos_modelos
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo de modelos '{file_name}'.")
        st.info("Por favor, ejecuta el script 'train_models.py' en tu terminal.")
        return None

# --- ¡NUEVA FUNCIÓN! Cargar datos climáticos ---
@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    """Carga los datos climáticos descargados."""
    try:
        df_clima = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha']).dt.date
        return df_clima
    except FileNotFoundError:
        st.warning(f"Advertencia: No se encontró '{file_name}'. El pronóstico funcionará sin datos climáticos.")
        return None

# --- Funciones de Predicción Avanzada ---
def simular_festivos(df_fechas):
    """
    Usa la librería 'holidays' para crear la feature 'es_festivo'.
    Asume que df_fechas tiene una columna 'ds'.
    """
    df = df_fechas.copy()
    # Festivos Reales (España), +1 año para predicciones
    festivos_espana = holidays.Spain(years=[2022, 2023, 2024, 2025])
    df['es_festivo'] = df['ds'].isin(festivos_espana).astype(int)
    return df

def crear_features_avanzadas_para_prediccion(df_diario, df_futuro, df_clima):
    """
    Crea las features (lag, rolling, clima, festivos) para el set de predicción.
    """
    df_diario_copy = df_diario.set_index('ds')
    df_futuro_copy = df_futuro.set_index('ds')
    
    df = pd.concat([df_diario_copy, df_futuro_copy])
    df = df.reset_index() # 'ds' es ahora una columna

    # 1. Características de tiempo
    df['mes'] = df['ds'].dt.month
    df['dia_del_ano'] = df['ds'].dt.dayofyear
    df['dia_de_la_semana'] = df['ds'].dt.dayofweek
    df['ano'] = df['ds'].dt.year
    
    # 2. Características Externas (Reales y Simuladas)
    df = simular_festivos(df) # <-- Pasa el df con la columna 'ds'
    df['temporada_gripe'] = df['mes'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df['temporada_polen'] = df['mes'].isin([3, 4, 5, 6]).astype(int)
    
    if df_clima is not None:
        df_clima_copy = df_clima.copy()
        df_clima_copy['ds'] = pd.to_datetime(df_clima_copy['Fecha'])
        df = df.merge(df_clima_copy[['ds', 'Temperatura_Media']], on='ds', how='left')
        df['Temperatura_Media'] = df['Temperatura_Media'].fillna(method='ffill').fillna(method='bfill')
    else:
        df['Temperatura_Media'] = 15.0 # Valor neutro
    
    # 3. Características de Retraso (Lag) y Móviles (Rolling)
    df = df.set_index('ds') # Volver a poner 'ds' como índice para shift/rolling
    df['ventas_lag_1'] = df['y'].shift(1)
    df['ventas_lag_7'] = df['y'].shift(7)
    df['media_movil_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean()
    df['media_movil_30d'] = df['y'].shift(1).rolling(window=30, min_periods=1).mean()
    
    df = df.bfill().reset_index() # Devolver 'ds' como columna
    
    return df.iloc[-len(df_futuro):]

def generar_pronostico_avanzado(model, df_historico, df_clima, dias_a_pronosticar, fecha_maxima_historica):
    """Genera la predicción de demanda usando el modelo AVANZADO."""
    fecha_max_dt = pd.to_datetime(fecha_maxima_historica)
    fechas_futuras = pd.date_range(start=fecha_max_dt + pd.Timedelta(days=1), periods=dias_a_pronosticar, freq='D')
    df_futuro_base = pd.DataFrame({'ds': fechas_futuras, 'y': np.nan})
    df_diario = df_historico.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    
    # --- ¡CORRECCIÓN! Pasar 'df_clima' a la función ---
    df_futuro_features = crear_features_avanzadas_para_prediccion(df_diario, df_futuro_base, df_clima)

    features = [
        'mes', 'dia_del_ano', 'dia_de_la_semana', 'ano', 'es_festivo',
        'temporada_gripe', 'temporada_polen', 'Temperatura_Media',
        'ventas_lag_1', 'ventas_lag_7',
        'media_movil_7d', 'media_movil_30d'
    ]
    
    df_futuro_features = df_futuro_features.fillna(0)
    X_pred = df_futuro_features[features]
    
    prediccion_futura = model.predict(X_pred)
    prediccion_futura[prediccion_futura < 0] = 0
    return prediccion_futura.round(0).astype(int) 

# --- INTERFAZ DE STREAMLIT ---
st.title("Sistema de Alerta y Priorización de Stock (IA)")

st.info("""
**¿Para qué sirve esto?**
Esta herramienta predice la demanda diaria de cada producto y la compara con el stock actual para **calcular los "Días hasta Rotura"**. 
Luego, prioriza automáticamente las alertas (Alta, Media, Baja) para que el equipo de operaciones sepa qué pedidos son más urgentes.
""", icon="ℹ️") 

df_total = cargar_datos()
datos_modelos = cargar_modelos()
df_clima = cargar_clima() # <-- ¡AÑADIR CARGA DE CLIMA!

if df_total is not None and datos_modelos is not None:
    df_stock_actual = simular_stock_actual(df_total)

    # --- Filtros de la Barra Lateral ---
    st.sidebar.title("Parámetros de Alerta")
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Selecciona Farmacia:", options=lista_farmacias, key='alerta_farmacia')
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    try:
        default_index = lista_categorias.index('Antigripal')
    except ValueError:
        default_index = 0 
    cat_sel = st.sidebar.selectbox("Selecciona Categoría:", options=lista_categorias, index=default_index, key='alerta_categoria')
    st.sidebar.divider()
    
    dias_a_pronosticar = st.sidebar.slider(
        "Horizonte de Análisis (Días):", 
        min_value=7, max_value=30, value=14, step=7,
        help="¿Con cuántos días de antelación quieres analizar el riesgo de rotura?"
    )
    
    stock_seguridad_pct = st.sidebar.slider(
        "Stock de Seguridad (% de Demanda):", 
        min_value=0, max_value=100, value=20, step=5,
        help="Porcentaje de la demanda total del horizonte que se guardará como 'colchón'. La rotura se calcula cuando se empieza a consumir este colchón."
    )
    
    PRIORIDAD_ALTA_DIAS = 3
    PRIORIDAD_MEDIA_DIAS = 7
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Leyenda de Prioridad**")
    
    st.sidebar.markdown(f"""
    <div style="padding: 6px; border-radius: 5px; background-color: {COLOR_ALTA}; color: white; margin-bottom: 5px; font-family: sans-serif; font-size: 0.9rem;">
        <strong>ALTA:</strong> Rotura en ≤ {PRIORIDAD_ALTA_DIAS} días
    </div>
    <div style="padding: 6px; border-radius: 5px; background-color: {COLOR_MEDIA}; color: white; margin-bottom: 5px; font-family: sans-serif; font-size: 0.9rem;">
        <strong>MEDIA:</strong> Rotura en ≤ {PRIORIDAD_MEDIA_DIAS} días
    </div>
    <div style="padding: 6px; border-radius: 5px; background-color: {COLOR_BAJA}; color: white; margin-bottom: 5px; font-family: sans-serif; font-size: 0.9rem;">
        <strong>BAJA:</strong> Rotura en > {PRIORIDAD_MEDIA_DIAS} días
    </div>
    <div style="padding: 6px; border-radius: 5px; background-color: #0E1117; color: #FAFAFA; border: 1px solid #333; margin-bottom: 5px; font-family: sans-serif; font-size: 0.9rem;">
        <strong>OK:</strong> Stock suficiente
    </div>
    """, unsafe_allow_html=True)


    if st.button("Generar Reporte de Priorización", type="primary", use_container_width=True):
        df_a_revisar = df_stock_actual.copy()
        if farmacia_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Farmacia_ID'] == farmacia_sel]
        df_a_revisar = df_a_revisar.merge(df_total[['Producto', 'Categoria']].drop_duplicates(), on='Producto', how='left')
        if cat_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Categoria'] == cat_sel]

        if df_a_revisar.empty:
            st.warning("No hay productos/farmacias que coincidan con los filtros.")
        else:
            fecha_max_hist = df_total['Fecha'].max()
            modelos_cargados = datos_modelos['modelos']
            resultados = []
            barra_progreso = st.progress(0, text="Iniciando análisis de priorización...")
            total_productos = len(df_a_revisar)
            
            modelos_sin_datos_count = 0
            
            for i, row in enumerate(df_a_revisar.itertuples()):
                farmacia_id = row.Farmacia_ID
                producto = row.Producto
                stock_actual = row.Stock_Actual
                barra_progreso.progress((i+1)/total_productos, text=f"Priorizando: {producto} en {farmacia_id}...")
                
                clave_modelo = f"{farmacia_id}::{producto}"
                info_modelo = modelos_cargados.get(clave_modelo)
                
                if info_modelo is None:
                    modelos_sin_datos_count += 1
                    continue 
                
                modelo = info_modelo['model']
                df_historico_producto = df_total[
                    (df_total['Farmacia_ID'] == farmacia_id) &
                    (df_total['Producto'] == producto)
                ]
                
                # --- ¡CORRECCIÓN! Pasar 'df_clima' a la función ---
                predicciones_diarias = generar_pronostico_avanzado(modelo, df_historico_producto, df_clima, dias_a_pronosticar, fecha_max_hist)
                demanda_predicha_total = predicciones_diarias.sum()
                
                stock_seguridad_unidades = demanda_predicha_total * (stock_seguridad_pct / 100)
                stock_util = stock_actual - stock_seguridad_unidades
                
                stock_restante = stock_util
                dias_hasta_rotura = "OK"
                
                for dia, demanda_dia in enumerate(predicciones_diarias):
                    stock_restante -= demanda_dia
                    if stock_restante <= 0:
                        dias_hasta_rotura = f"{dia + 1} días"
                        break 
                
                if dias_hasta_rotura == "OK":
                    prioridad = "OK"
                else:
                    num_dias = int(dias_hasta_rotura.split(" ")[0])
                    if num_dias <= PRIORIDAD_ALTA_DIAS:
                        prioridad = "Alta"
                    elif num_dias <= PRIORIDAD_MEDIA_DIAS:
                        prioridad = "Media"
                    else:
                        prioridad = "Baja"
                
                resultados.append({
                    "Farmacia": farmacia_id, "Producto": producto, "Prioridad": prioridad,
                    "Días hasta Rotura": dias_hasta_rotura, "Stock Actual": stock_actual,
                    "Stock de Seguridad": int(stock_seguridad_unidades), "Stock Útil": int(stock_util),
                    f"Demanda Predicha ({dias_a_pronosticar} días)": demanda_predicha_total,
                })
            
            barra_progreso.empty() 
            st.success("¡Análisis de priorización completado!")
            
            if modelos_sin_datos_count > 0:
                st.info(f"{modelos_sin_datos_count} productos fueron omitidos del reporte por no tener suficientes datos históricos para un modelo de IA.")
            
            df_resultados = pd.DataFrame(resultados)
            
            if df_resultados.empty:
                st.warning("No se generaron resultados de alerta para los filtros seleccionados (excluyendo productos sin datos).")
            else:
                columnas_resumen = ["Farmacia", "Producto", "Prioridad", "Días hasta Rotura"]
                df_resumen = df_resultados[columnas_resumen]
                
                columnas_detalle = [
                    "Farmacia", "Producto", "Prioridad", "Días hasta Rotura", 
                    "Stock Actual", "Stock de Seguridad", "Stock Útil", 
                    f"Demanda Predicha ({dias_a_pronosticar} días)"
                ]
                df_detalle = df_resultados[columnas_detalle]
                
                def estilizar_prioridad(fila):
                    color = ""
                    if fila.Prioridad == "Alta": color = COLOR_ALTA
                    elif fila.Prioridad == "Media": color = COLOR_MEDIA
                    elif fila.Prioridad == "Baja": color = COLOR_BAJA
                    
                    if color:
                        return [f'background-color: {color}; color: white'] * len(fila)
                    else:
                        return [''] * len(fila)

                st.header("Lista de Prioridades (Resumen)")
                st.dataframe(df_resumen.style.apply(estilizar_prioridad, axis=1), use_container_width=True)
                
                with st.expander("Ver cálculos y detalles completos"):
                    st.dataframe(df_detalle, use_container_width=True)
                
                csv_data = convert_df_to_csv(df_detalle)
                st.download_button(
                    label="Descargar Reporte Detallado en CSV",
                    data=csv_data,
                    file_name=f"reporte_priorizacion_detallado.csv",
                    mime='text/csv',
                    use_container_width=True
                )
else:
    st.error("Error al cargar los datos. Revisa el archivo CSV y asegúrate de haber ejecutado 'train_models.py'.")