import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Alerta de Stock (IA)", layout="wide")

# --- 1. Paleta de Colores Profesional (Con Azul Suave) ---
COLOR_ALTA = "#9B2B2B"     # Rojo Ladrillo (Dark Red)
COLOR_MEDIA = "#B9770E"    # Ámbar Oscuro (Dark Amber)
COLOR_BAJA = "#4682B4"     # Azul Acero (SteelBlue) <-- ¡CAMBIO A AZUL!
# Las filas "OK" se quedarán en blanco (sin color)

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

# --- Funciones de Predicción Avanzada ---
def simular_festivos(df_fechas):
    """Simula días festivos para los datos de predicción."""
    df = df_fechas.copy()
    df['dia_del_ano'] = df['ds'].dt.dayofyear
    df['dia_de_la_semana'] = df['ds'].dt.dayofweek
    festivos_fijos = [1, 121, 359] 
    es_puente = (df['dia_de_la_semana'] == 4) & (df['dia_del_ano'] % 100 == 0)
    df['es_festivo'] = df['dia_del_ano'].isin(festivos_fijos) | es_puente
    df['es_festivo'] = df['es_festivo'].astype(int)
    return df

def crear_features_avanzadas_para_prediccion(df_diario, df_futuro):
    """Crea las features (lag, rolling) para el set de predicción."""
    df_diario_copy = df_diario.set_index('ds')
    df_futuro_copy = df_futuro.set_index('ds')
    df = pd.concat([df_diario_copy, df_futuro_copy])
    df['mes'] = df.index.month
    df['dia_del_ano'] = df.index.dayofyear
    df['dia_de_la_semana'] = df.index.dayofweek
    df['ano'] = df.index.year
    df = simular_festivos(df.reset_index()).set_index('ds')
    df['ventas_lag_1'] = df['y'].shift(1)
    df['ventas_lag_7'] = df['y'].shift(7)
    df['media_movil_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean()
    df['media_movil_30d'] = df['y'].shift(1).rolling(window=30, min_periods=1).mean()
    df = df.bfill()
    return df.iloc[-len(df_futuro):]

def generar_pronostico_avanzado(model, df_historico, dias_a_pronosticar, fecha_maxima_historica):
    """Genera la predicción de demanda usando el modelo AVANZADO."""
    fecha_max_dt = pd.to_datetime(fecha_maxima_historica)
    fechas_futuras = pd.date_range(start=fecha_max_dt + pd.Timedelta(days=1), periods=dias_a_pronosticar, freq='D')
    df_futuro_base = pd.DataFrame({'ds': fechas_futuras, 'y': np.nan})
    df_diario = df_historico.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    df_futuro_features = crear_features_avanzadas_para_prediccion(df_diario, df_futuro_base)
    features = ['mes', 'dia_del_ano', 'dia_de_la_semana', 'ano', 'es_festivo', 'ventas_lag_1', 'ventas_lag_7', 'media_movil_7d', 'media_movil_30d']
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
    
    # --- 3. Leyenda de Colores Limpia y Profesional ---
    st.sidebar.markdown("**Leyenda de Prioridad**") # Título más pequeño
    
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
    # --- FIN DE MEJORA DE LEYENDA ---

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
                
                predicciones_diarias = generar_pronostico_avanzado(modelo, df_historico_producto, dias_a_pronosticar, fecha_max_hist)
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
                columnas_ordenadas = [
                    "Farmacia", "Producto", "Prioridad", "Días hasta Rotura", 
                    "Stock Actual", "Stock de Seguridad", "Stock Útil", 
                    f"Demanda Predicha ({dias_a_pronosticar} días)"
                ]
                df_resultados = df_resultados[columnas_ordenadas]

                def estilizar_prioridad(fila):
                    color = ""
                    if fila.Prioridad == "Alta":
                        color = COLOR_ALTA
                    elif fila.Prioridad == "Media":
                        color = COLOR_MEDIA
                    elif fila.Prioridad == "Baja":
                        color = COLOR_BAJA
                    
                    if color:
                        # Aplicamos el color solo a la fila, pero dejamos el texto blanco/claro
                        return [f'background-color: {color}; color: white'] * len(fila)
                    else:
                        # Devuelve vacío para las filas "OK" (usa el tema por defecto)
                        return [''] * len(fila)

                st.dataframe(df_resultados.style.apply(estilizar_prioridad, axis=1), use_container_width=True)
                
                csv_data = convert_df_to_csv(df_resultados)
                st.download_button(
                    label="Descargar Reporte de Priorización en CSV",
                    data=csv_data,
                    file_name=f"reporte_priorizacion_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
else:
    st.error("Error al cargar los datos. Revisa el archivo CSV y asegúrate de haber ejecutado 'train_models.py'.")