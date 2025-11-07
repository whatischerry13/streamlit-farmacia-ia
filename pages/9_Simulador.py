import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
from datetime import datetime, timedelta
import altair as alt

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Configuración de Página ---
st.set_page_config(page_title="Simulador de Escenarios", layout="wide")

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
st.title("Simulador de Escenarios 'What-If'")
st.info("""
**¿Para qué sirve esto?**
Esta es una herramienta de **análisis prescriptivo**. Permite simular el impacto de eventos de negocio (como campañas de marketing) sobre el inventario *antes* de que ocurran. 
Responde a preguntas como: *"¿Qué pasaría si lanzo una campaña 2x1 y la demanda de antigripales sube un 50%? ¿Qué farmacias se quedarían sin stock y cuándo?"*
""", icon="ℹ️")

df_total = cargar_datos()
datos_modelos = cargar_modelos()

if df_total is not None and datos_modelos is not None:
    df_stock_actual = simular_stock_actual(df_total)

    # --- Filtros de la Barra Lateral ---
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
    st.sidebar.header("Parámetros de Análisis")
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Selecciona Farmacia:", options=lista_farmacias, key='sim_farmacia')
    
    dias_a_pronosticar = st.sidebar.slider(
        "Horizonte de Simulación (Días):", 
        min_value=7, max_value=30, value=14, step=7, key='sim_dias'
    )
    stock_seguridad_pct = st.sidebar.slider(
        "Stock de Seguridad (% de Demanda):", 
        min_value=0, max_value=100, value=20, step=5, key='sim_seguridad',
        help="Colchón de seguridad. La simulación calculará la rotura cuando el stock caiga por debajo de este colchón."
    )

    if st.button("Ejecutar Simulación", type="primary", use_container_width=True):
        df_a_revisar = df_stock_actual.copy()
        if farmacia_sel != 'Todas':
            df_a_revisar = df_a_revisar[df_a_revisar['Farmacia_ID'] == farmacia_sel]
        
        df_precios = df_total[['Producto', 'Precio_Unitario_€']].drop_duplicates(subset='Producto').set_index('Producto')
        df_a_revisar = df_a_revisar.merge(
            df_total[['Producto', 'Categoria']].drop_duplicates(),
            on='Producto', how='left'
        )
        df_a_revisar = df_a_revisar[df_a_revisar['Categoria'].isin(['Alergia', 'Antigripal'])]

        if df_a_revisar.empty:
            st.warning("No hay productos que coincidan con los filtros.")
        else:
            fecha_max_hist = df_total['Fecha'].max()
            modelos_cargados = datos_modelos['modelos']
            
            resultados_tabla = []
            df_grafico_list = [] 
            ventas_en_riesgo_total = 0
            productos_en_riesgo_count = 0
            dias_hasta_rotura_list = []
            
            barra_progreso = st.progress(0, text="Iniciando simulación avanzada...")
            total_productos = len(df_a_revisar)
            
            for i, row in enumerate(df_a_revisar.itertuples()):
                farmacia_id = row.Farmacia_ID
                producto = row.Producto
                categoria = row.Categoria
                stock_actual = row.Stock_Actual
                try:
                    precio_unitario = df_precios.loc[producto, 'Precio_Unitario_€']
                except KeyError:
                    precio_unitario = 0 
                
                barra_progreso.progress((i+1)/total_productos, text=f"Simulando: {producto}...")
                
                clave_modelo = f"{farmacia_id}::{producto}"
                info_modelo = modelos_cargados.get(clave_modelo)
                
                if info_modelo is None:
                    continue 
                
                modelo = info_modelo['model']
                df_historico_producto = df_total[
                    (df_total['Farmacia_ID'] == farmacia_id) &
                    (df_total['Producto'] == producto)
                ]
                
                predicciones_diarias = generar_pronostico_avanzado(modelo, df_historico_producto, dias_a_pronosticar, fecha_max_hist)
                
                ajuste_pct = 0
                if categoria == 'Antigripal': ajuste_pct = ajuste_antigripal
                elif categoria == 'Alergia': ajuste_pct = ajuste_alergia
                multiplicador = 1 + (ajuste_pct / 100)
                
                demanda_simulada_diaria = (predicciones_diarias * multiplicador).round(0).astype(int)
                demanda_total_simulada = demanda_simulada_diaria.sum()
                
                stock_seguridad_unidades = demanda_total_simulada * (stock_seguridad_pct / 100)
                stock_util = stock_actual - stock_seguridad_unidades
                
                stock_restante = stock_util
                dias_hasta_rotura = "OK"
                ventas_perdidas_producto = 0
                
                df_producto_grafico = pd.DataFrame({
                    'Dia': range(1, dias_a_pronosticar + 1),
                    'Stock_Restante': 0.0,
                    'Producto_Farmacia': f"{producto} ({farmacia_id})"
                })
                
                for dia in range(dias_a_pronosticar):
                    demanda_dia = demanda_simulada_diaria[dia]
                    if stock_restante > 0:
                        unidades_vendidas = min(stock_restante, demanda_dia)
                        stock_restante -= unidades_vendidas
                        ventas_perdidas_dia = demanda_dia - unidades_vendidas
                    else:
                        stock_restante = 0.0
                        ventas_perdidas_dia = demanda_dia
                    
                    df_producto_grafico.loc[dia, 'Stock_Restante'] = stock_restante
                    ventas_perdidas_producto += (ventas_perdidas_dia * precio_unitario)
                    
                    if dias_hasta_rotura == "OK" and stock_restante <= 0:
                        dias_hasta_rotura = f"{dia + 1} días"

                if dias_hasta_rotura != "OK":
                    productos_en_riesgo_count += 1
                    ventas_en_riesgo_total += ventas_perdidas_producto
                    dias_hasta_rotura_list.append(int(dias_hasta_rotura.split(" ")[0]))
                    df_grafico_list.append(df_producto_grafico)
                
                resultados_tabla.append({
                    "Farmacia": farmacia_id, "Producto": producto,
                    "Días hasta Rotura": dias_hasta_rotura,
                    "Stock Actual": stock_actual, "Stock Útil": int(stock_util),
                    "Demanda Simulada Total": demanda_total_simulada,
                    "Ventas en Riesgo (€)": f"{ventas_perdidas_producto:,.2f} €",
                    "Ajuste Aplicado (%)": ajuste_pct
                })
            
            barra_progreso.empty() 
            st.success(f"¡Simulación completada!")
            
            st.header("Resultados de la Simulación")
            
            # --- MEJORA 1: Explicar "Desde qué día partimos" ---
            fecha_inicio_sim = (pd.to_datetime(fecha_max_hist) + timedelta(days=1)).strftime('%d-%b-%Y')
            fecha_fin_sim = (pd.to_datetime(fecha_max_hist) + timedelta(days=dias_a_pronosticar)).strftime('%d-%b-%Y')
            st.subheader(f"Horizonte: {fecha_inicio_sim} al {fecha_fin_sim}")
            # --- FIN MEJORA 1 ---
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Productos en Riesgo de Rotura", f"{productos_en_riesgo_count}")
            col2.metric("Ventas Totales en Riesgo", f"{ventas_en_riesgo_total:,.0f} €")
            if dias_hasta_rotura_list:
                dias_avg = np.mean(dias_hasta_rotura_list)
                col3.metric("Días Promedio hasta Rotura", f"{dias_avg:.1f} días")
            else:
                col3.metric("Días Promedio hasta Rotura", "N/A")
            
            st.divider()

            # --- MEJORA 2: Título y explicación del gráfico ---
            st.subheader(f"Proyección de Agotamiento de Stock (Top 5 en Riesgo)")
            
            if not df_grafico_list:
                st.info("¡Buenas noticias! Ningún producto está en riesgo de rotura de stock bajo este escenario.")
            else:
                df_grafico_final = pd.concat(df_grafico_list)
                
                df_ranking = pd.DataFrame(resultados_tabla)
                df_ranking = df_ranking[df_ranking['Días hasta Rotura'] != 'OK']
                df_ranking['Dias_Num'] = df_ranking['Días hasta Rotura'].str.replace(' días', '').astype(int)
                df_ranking['Producto_Farmacia'] = df_ranking.apply(lambda row: f"{row['Producto']} ({row['Farmacia']})", axis=1)
                top_5_en_riesgo = df_ranking.nsmallest(5, 'Dias_Num')['Producto_Farmacia']

                df_grafico_final = df_grafico_final[df_grafico_final['Producto_Farmacia'].isin(top_5_en_riesgo)]
                
                # Línea de Rotura de Stock (Y=0)
                regla_cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
                
                # Gráfico de líneas
                line_chart = alt.Chart(df_grafico_final).mark_line(point=True).encode(
                    x=alt.X('Dia', title=f'Días desde {fecha_inicio_sim}'), # Eje X más claro
                    y=alt.Y('Stock_Restante', title='Unidades de Stock (Útil)'), # Eje Y más claro
                    color=alt.Color('Producto_Farmacia', title='Producto (Farmacia)'),
                    tooltip=['Producto_Farmacia', 'Dia', 'Stock_Restante']
                ).interactive()
                
                st.altair_chart(line_chart + regla_cero, use_container_width=True)
                
                st.info("""
                **Cómo leer este gráfico:**
                * Este gráfico muestra cómo se agota el **"Stock Útil"** (Stock Actual - Stock de Seguridad) día a día.
                * La **línea roja (en 0)** es la rotura de stock.
                * Cuando una línea de producto la cruza, significa que ese día se agotan las existencias.
                """)
            # --- FIN MEJORA 2 ---

            # --- MEJORA 3: Tabla Limpia + Expander de Detalles ---
            st.subheader("Resumen de Impacto por Producto")
            
            df_resultados = pd.DataFrame(resultados_tabla)
            
            # Columnas clave para el resumen
            columnas_resumen = ["Farmacia", "Producto", "Días hasta Rotura", "Ventas en Riesgo (€)", "Ajuste Aplicado (%)"]
            df_resumen = df_resultados[columnas_resumen]
            
            # Estilo para la tabla resumen
            def estilizar_riesgo(fila):
                if fila["Días hasta Rotura"] != "OK":
                    return [f'background-color: #9B2B2B; color: white'] * len(fila) # Rojo oscuro
                else:
                    return [''] * len(fila)

            st.dataframe(df_resumen.style.apply(estilizar_riesgo, axis=1), use_container_width=True)

            with st.expander("Ver cálculos y detalles completos"):
                st.dataframe(df_resultados, use_container_width=True)
            
            # El botón de descarga seguirá descargando la tabla detallada
            csv_data = convert_df_to_csv(df_resultados)
            st.download_button(
                label="Descargar Simulación Detallada en CSV",
                data=csv_data,
                file_name=f"reporte_simulacion.csv",
                mime='text/csv',
                use_container_width=True
            )
else:
    st.error("Error al cargar los datos. Revisa el archivo CSV y asegúrate de haber ejecutado 'train_models.py'.")