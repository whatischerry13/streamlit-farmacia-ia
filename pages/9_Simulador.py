import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
import warnings
import joblib
from datetime import datetime

# --- MEJORA VISUAL: Configuración de Pestaña ---
st.set_page_config(page_title="Simulador 'What-If'", page_icon="🧪", layout="wide")

# --- MEJORA FUNCIONAL: Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- FUNCIONES DE DATOS Y MODELOS ---
@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

@st.cache_data
def simular_stock_actual(_df_total):
    st.info("Simulando inventario de stock actual... (se ejecuta una vez)", icon="📦")
    df_ventas_diarias = _df_total.groupby(['Farmacia_ID', 'Producto'])['Cantidad'].mean().reset_index()
    df_ventas_diarias = df_ventas_diarias.rename(columns={'Cantidad': 'Venta_Media_Diaria'})
    np.random.seed(123)
    dias_stock_simulados = np.random.randint(1, 21, size=len(df_ventas_diarias))
    df_ventas_diarias['Stock_Actual'] = (df_ventas_diarias['Venta_Media_Diaria'] * dias_stock_simulados).round(0).astype(int)
    return df_ventas_diarias[['Farmacia_ID', 'Producto', 'Stock_Actual']]

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        datos_modelos = joblib.load(file_name)
        st.success(f"Modelos de IA cargados (entrenados el {datos_modelos['fecha_entrenamiento'].strftime('%Y-%m-%d %H:%M')})")
        return datos_modelos
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        st.info("Por favor, ejecuta el script 'train_models.py' en tu terminal para generar el archivo de modelos.")
        return None

def crear_caracteristicas_temporales(df_in):
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
    fecha_max_dt = pd.to_datetime(fecha_maxima_historica)
    fechas_futuras = pd.date_range(start=fecha_max_dt + pd.Timedelta(days=1), periods=dias_a_pronosticar, freq='D')
    df_futuro = pd.DataFrame({'ds': fechas_futuras})
    df_futuro_preparado = crear_caracteristicas_temporales(df_futuro)
    features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
    prediccion_futura = model.predict(df_futuro_preparado[features])
    prediccion_futura[prediccion_futura < 0] = 0
    return prediccion_futura.sum().round(0).astype(int)

# --- INTERFAZ DE STREAMLIT ---
st.title("🧪 Simulador de Escenarios 'What-If'")
st.info("""
**💡 ¿Para qué sirve esto?**
Esta es una herramienta de **decisión estratégica**. Te permite probar escenarios de negocio y ver su impacto en el inventario *antes* de que ocurran. 
Responde a preguntas como: *"¿Qué pasaría si lanzo una campaña 2x1 y la demanda de antigripales sube un 50%? ¿Qué farmacias se quedarían sin stock y cuándo?"*
""", icon="💡")
st.markdown("### Simula el impacto de campañas de marketing o cambios en la demanda.")

df_total = cargar_datos()
datos_modelos = cargar_modelos()

if df_total is not None and datos_modelos is not None:
    df_stock_actual = simular_stock_actual(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Parámetros del Simulador")
    st.sidebar.header("Ajuste de Escenarios (What-If)")
    st.sidebar.info("Ajusta la demanda esperada para probar escenarios.")

    ajuste_antigripal = st.sidebar.slider(
        "Ajuste Demanda 'Antigripal' (%)", -50, 100, 0, 10,
        help="Simula un aumento (ej. +50% por campaña) o caída (ej. -20% por invierno suave) en la demanda."
    )
    ajuste_alergia = st.sidebar.slider(
        "Ajuste Demanda 'Alergia' (%)", -50, 100, 0, 10,
        help="Simula el impacto de una primavera fuerte (+30%) o débil (-10%)."
    )
    st.sidebar.divider()
    st.sidebar.header("Parámetros de Alerta")
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Selecciona Farmacia:", options=lista_farmacias, key='sim_farmacia')
    dias_a_pronosticar = st.sidebar.slider("Horizonte de Simulación (Días):", 1, 30, 7, key='sim_dias')
    stock_seguridad_pct = st.sidebar.slider("Stock de Seguridad (%):", 0, 100, 20, key='sim_seguridad')

    # --- LÓGICA DE EJECUCIÓN ---
    
    # Creamos un placeholder para la tabla de resultados
    results_container = st.container()
    
    if st.button("▶️ Ejecutar Simulación", type="primary", use_container_width=True):
        df_a_revisar = df_stock_actual.copy()
        if farmacia_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Farmacia_ID'] == farmacia_sel]
        df_a_revisar = df_a_revisar.merge(df_total[['Producto', 'Categoria']].drop_duplicates(), on='Producto', how='left')
        df_a_revisar = df_a_revisar[df_a_revisar['Categoria'].isin(['Alergia', 'Antigripal'])]

        if df_a_revisar.empty:
            st.warning("No hay productos que coincidan con los filtros.")
        else:
            fecha_max_hist = df_total['Fecha'].max()
            modelos_cargados = datos_modelos['modelos']
            resultados = []
            barra_progreso = st.progress(0, text="Iniciando simulación...")
            total_productos = len(df_a_revisar)
            
            for i, row in enumerate(df_a_revisar.itertuples()):
                farmacia_id = row.Farmacia_ID
                producto = row.Producto
                categoria = row.Categoria
                stock_actual = row.Stock_Actual
                barra_progreso.progress((i+1)/total_productos, text=f"Simulando: {producto}...")
                clave_modelo = f"{farmacia_id}::{producto}"
                modelo = modelos_cargados.get(clave_modelo)
                
                if modelo is None:
                    demanda_predicha = 0; alerta = "Sin Datos (IA)"; stock_necesario = 0; demanda_ajustada = 0
                else:
                    demanda_predicha = generar_pronostico(modelo, dias_a_pronosticar, fecha_max_hist)
                    ajuste_pct = 0
                    if categoria == 'Antigripal': ajuste_pct = ajuste_antigripal
                    elif categoria == 'Alergia': ajuste_pct = ajuste_alergia
                    multiplicador = 1 + (ajuste_pct / 100)
                    demanda_ajustada = demanda_predicha * multiplicador
                    stock_seguridad_unidades = demanda_ajustada * (stock_seguridad_pct / 100)
                    stock_necesario = demanda_ajustada + stock_seguridad_unidades
                    if stock_actual < stock_necesario: alerta = "ALERTA (Pedir)"
                    else: alerta = "OK"
                
                resultados.append({
                    "Farmacia": farmacia_id, "Producto": producto, "Stock Actual": stock_actual,
                    "Demanda IA (Base)": int(demanda_predicha), f"Ajuste ({ajuste_pct}%)": int(demanda_ajustada - demanda_predicha),
                    "Demanda Simulada": int(demanda_ajustada), "Stock Necesario": int(stock_necesario),
                    "Estado Simulado": alerta
                })
            
            barra_progreso.empty() 
            
            # Escribir los resultados en el contenedor
            with results_container:
                st.success(f"¡Simulación completada para el escenario: Alergia ({ajuste_alergia}%), Antigripal ({ajuste_antigripal}%)!")
                df_resultados = pd.DataFrame(resultados)
                
                def estilizar_alertas(fila):
                    if fila['Estado Simulado'] == "ALERTA (Pedir)": return ['background-color: #FF4B4B; color: white'] * len(fila)
                    elif fila['Estado Simulado'] == "Sin Datos (IA)": return ['background-color: #A9A9A9; color: white'] * len(fila)
                    else: return [''] * len(fila)

                st.dataframe(df_resultados.style.apply(estilizar_alertas, axis=1), use_container_width=True)
                
                # --- MEJORA: BOTÓN DE DESCARGA ---
                csv_data = convert_df_to_csv(df_resultados)
                st.download_button(
                    label="Descargar Simulación en CSV",
                    data=csv_data,
                    file_name=f"reporte_simulacion_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
else:
    st.error("Error al cargar los datos. Revisa el archivo CSV y asegúrate de haber ejecutado 'train_models.py'.")