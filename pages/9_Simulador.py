import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
import holidays
import altair as alt
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Configuración de Página
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
        st.error(f"Error: No se encontró el archivo '{file_name}'.")
        return None

@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    """Carga los datos climáticos."""
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except:
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
        st.error(f"Error: No se encontró el archivo de modelos '{file_name}'.")
        st.info("Por favor, ejecuta el script 'train_models.py' en tu terminal.")
        return None

# --- LÓGICA DE PREDICCIÓN 

def simular_festivos(df_fechas):
    """Simula días festivos para los datos de predicción."""
    # Esta función auxiliar ya no es estrictamente necesaria dentro de 'crear_features_un_paso'
    # porque calculamos los festivos directamente allí, pero la dejo por compatibilidad.
    return df_fechas

# 2. Generador de Features paso a paso (Coincide con el modelo Premium)
def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    """Genera las 16 features premium para un solo día futuro."""
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    
    # A. Ciclos Temporales
    mes = fecha_obj.month
    dia_sem = fecha_obj.weekday()
    row['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    row['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    
    # B. Festivos y Temporadas
    es_holidays = holidays.Spain(years=[fecha_obj.year])
    row['es_festivo'] = int(fecha_obj in es_holidays)
    row['temp_gripe'] = int(mes in [10, 11, 12, 1, 2])
    row['temp_alergia'] = int(mes in [3, 4, 5, 6])
    
    # C. Clima
    t_media = 15.0
    if df_clima is not None:
        match = df_clima[df_clima['Fecha'] == fecha_obj]
        if not match.empty: t_media = match.iloc[0]['Temperatura_Media']
    row['Temperatura_Media'] = t_media
    
    # D. Lags y Rolling (Usando el historial acumulado)
    vals = historia_y
    row['lag_1'] = vals[-1] if len(vals) >= 1 else 0
    row['lag_2'] = vals[-2] if len(vals) >= 2 else 0
    row['lag_7'] = vals[-7] if len(vals) >= 7 else 0
    row['lag_14'] = vals[-14] if len(vals) >= 14 else 0
    
    row['roll_mean_7'] = pd.Series(vals).rolling(7).mean().iloc[-1] if len(vals)>=7 else 0
    row['roll_mean_28'] = pd.Series(vals).rolling(28).mean().iloc[-1] if len(vals)>=28 else 0
    row['roll_std_7'] = pd.Series(vals).rolling(7).std().iloc[-1] if len(vals)>=7 else 0
    
    # Tendencia
    rm7_prev = pd.Series(vals).rolling(7).mean().iloc[-8] if len(vals)>=8 else row['roll_mean_7'].values[0]
    row['tendencia_semanal'] = row['roll_mean_7'] - rm7_prev
    
    # Orden exacto de columnas que espera el modelo Premium (16 features)
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_recursivo(model, df_hist_prod, df_clima, dias_futuros, fecha_max):
    """Bucle que predice día a día alimentándose a sí mismo."""
    # Extraemos solo la serie de cantidad ordenada
    historia = list(df_hist_prod.sort_values('Fecha')['Cantidad'].values)
    predicciones = []
    
    for i in range(dias_futuros):
        fecha_futura = fecha_max + timedelta(days=i+1)
        
        # Crear features para mañana usando la historia (que incluye predicciones pasadas)
        X_test = crear_features_un_paso(historia, fecha_futura, df_clima)
        
        # Predecir
        y_pred = max(0, model.predict(X_test)[0])
        
        # Guardar y actualizar historia para el siguiente ciclo del bucle
        predicciones.append(y_pred)
        historia.append(y_pred)
        
    return np.array(predicciones)

# --- INTERFAZ DE STREAMLIT ---
st.title("Simulador de Escenarios 'What-If'")
st.info("""
**Herramienta de Análisis Prescriptivo**
Permite simular el impacto de eventos de negocio (como campañas de marketing) sobre el inventario antes de que ocurran. 
Responde a preguntas como: "¿Qué pasaría si lanzo una campaña 2x1 y la demanda de antigripales sube un 50%? ¿Qué farmacias se quedarían sin stock y cuándo?"
""")

df_total = cargar_datos()
datos_modelos = cargar_modelos()
df_clima = cargar_clima()

if df_total is not None and datos_modelos is not None:
    df_stock_actual = simular_stock_actual(df_total)

    # --- Filtros ---
    st.sidebar.title("Parámetros del Simulador")
    st.sidebar.header("Ajuste de Escenarios")
    st.sidebar.info("Ajusta la demanda esperada para probar escenarios.")

    ajuste_antigripal = st.sidebar.slider(
        "Ajuste Demanda 'Antigripal' (%)", -50, 100, 0, 10,
        help="Simula un aumento o caída porcentual en la demanda."
    )
    ajuste_alergia = st.sidebar.slider(
        "Ajuste Demanda 'Alergia' (%)", -50, 100, 0, 10,
        help="Simula un aumento o caída porcentual en la demanda."
    )
    st.sidebar.divider()
    st.sidebar.header("Parámetros de Análisis")
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Selecciona Farmacia:", options=lista_farmacias, key='sim_farmacia')
    
    dias_a_pronosticar = st.sidebar.slider(
        "Horizonte de Simulación (Días):", 
        min_value=7, max_value=60, value=14, step=7, key='sim_dias'
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
        
        # Precios para calcular impacto económico
        df_precios = df_total[['Producto', 'Precio_Unitario_€']].drop_duplicates(subset='Producto').set_index('Producto')
        
        # Unir categorías
        df_a_revisar = df_a_revisar.merge(
            df_total[['Producto', 'Categoria']].drop_duplicates(),
            on='Producto', how='left'
        )
        # Filtrar solo productos modelables
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
                
                try: precio_unitario = df_precios.loc[producto, 'Precio_Unitario_€']
                except: precio_unitario = 0 
                
                barra_progreso.progress((i+1)/total_productos, text=f"Simulando: {producto}...")
                
                # Buscar modelo
                clave_modelo = f"{farmacia_id}::{producto}"
                info_modelo = modelos_cargados.get(clave_modelo)
                
                if info_modelo is None: continue 
                
                modelo = info_modelo['model']
                
                # Obtener historial específico para la recursión
                df_hist_prod = df_total[
                    (df_total['Farmacia_ID'] == farmacia_id) &
                    (df_total['Producto'] == producto)
                ]
                
                # --- PREDICCIÓN RECURSIVA 
                predicciones_diarias = predecir_recursivo(modelo, df_hist_prod, df_clima, dias_a_pronosticar, fecha_max_hist)
                
                # Aplicar Ajuste del Simulador
                ajuste_pct = 0
                if categoria == 'Antigripal': ajuste_pct = ajuste_antigripal
                elif categoria == 'Alergia': ajuste_pct = ajuste_alergia
                multiplicador = 1 + (ajuste_pct / 100)
                
                demanda_simulada_diaria = (predicciones_diarias * multiplicador).round(0).astype(int)
                demanda_total_simulada = demanda_simulada_diaria.sum()
                
                stock_seguridad_unidades = demanda_total_simulada * (stock_seguridad_pct / 100)
                stock_util = stock_actual - stock_seguridad_unidades
                
                # Calcular evolución de stock día a día
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
            st.success(f"Simulación completada")
            
            st.header("Resultados de la Simulación")
            
            # Fechas para el título
            fecha_inicio_sim = (pd.to_datetime(fecha_max_hist) + timedelta(days=1)).strftime('%d-%b-%Y')
            fecha_fin_sim = (pd.to_datetime(fecha_max_hist) + timedelta(days=dias_a_pronosticar)).strftime('%d-%b-%Y')
            st.subheader(f"Horizonte Simulado: {fecha_inicio_sim} al {fecha_fin_sim}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Productos en Riesgo", f"{productos_en_riesgo_count}")
            col2.metric("Ventas en Riesgo", f"{ventas_en_riesgo_total:,.0f} €")
            if dias_hasta_rotura_list:
                dias_avg = np.mean(dias_hasta_rotura_list)
                col3.metric("Promedio Días a Rotura", f"{dias_avg:.1f} días")
            else:
                col3.metric("Promedio Días a Rotura", "N/A")
            
            st.divider()

            st.subheader("Proyección de Agotamiento de Stock (Top 5 en Riesgo)")
            
            if not df_grafico_list:
                st.success("¡Buenas noticias! El inventario soporta este escenario sin roturas.")
            else:
                df_grafico_final = pd.concat(df_grafico_list)
                
                # Filtrar Top 5 peores casos
                df_ranking = pd.DataFrame(resultados_tabla)
                df_ranking = df_ranking[df_ranking['Días hasta Rotura'] != 'OK']
                df_ranking['Dias_Num'] = df_ranking['Días hasta Rotura'].str.replace(' días', '').astype(int)
                df_ranking['Producto_Farmacia'] = df_ranking.apply(lambda row: f"{row['Producto']} ({row['Farmacia']})", axis=1)
                top_5 = df_ranking.nsmallest(5, 'Dias_Num')['Producto_Farmacia']

                df_grafico_final = df_grafico_final[df_grafico_final['Producto_Farmacia'].isin(top_5)]
                
                # Gráfico
                regla_cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
                
                line_chart = alt.Chart(df_grafico_final).mark_line(point=True).encode(
                    x=alt.X('Dia', title=f'Días desde {fecha_inicio_sim}'), 
                    y=alt.Y('Stock_Restante', title='Stock Útil Restante (Unidades)'),
                    color=alt.Color('Producto_Farmacia', title='Producto (Farmacia)'),
                    tooltip=['Producto_Farmacia', 'Dia', 'Stock_Restante']
                ).interactive()
                
                st.altair_chart(line_chart + regla_cero, use_container_width=True)
                
                st.info("""
                **Cómo leer este gráfico:**
                * Muestra el agotamiento del **"Stock Útil"** día a día.
                * La **línea roja (0)** indica rotura de stock.
                * El cruce de una línea con el 0 indica el día exacto del agotamiento.
                """)

            st.subheader("Resumen de Impacto")
            
            df_resultados = pd.DataFrame(resultados_tabla)
            
            # Tabla Resumen Limpia
            cols_resumen = ["Farmacia", "Producto", "Días hasta Rotura", "Ventas en Riesgo (€)", "Ajuste Aplicado (%)"]
            df_resumen = df_resultados[cols_resumen]
            
            def estilo_riesgo(fila):
                if fila["Días hasta Rotura"] != "OK":
                    return [f'background-color: #8B0000; color: white'] * len(fila) # Rojo oscuro profesional
                else:
                    return [''] * len(fila)

            st.dataframe(df_resumen.style.apply(estilo_riesgo, axis=1), use_container_width=True)

            with st.expander("Ver cálculos detallados"):
                st.dataframe(df_resultados, use_container_width=True)
            
            csv_data = convert_df_to_csv(df_resultados)
            st.download_button(
                label="Descargar Simulación en CSV",
                data=csv_data,
                file_name=f"reporte_simulacion.csv",
                mime='text/csv',
                use_container_width=True
            )
else:
    st.error("Error al cargar datos. Revisa que 'train_models.py' se haya ejecutado correctamente.")