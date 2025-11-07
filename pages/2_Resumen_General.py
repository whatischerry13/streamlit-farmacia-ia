import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
import warnings
import joblib
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import holidays # <-- Importación necesaria

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=st.errors.StreamlitAPIWarning)

# --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN 1!! ---
# Añadimos la configuración de página para que use el layout ancho
st.set_page_config(page_title="Resumen General", layout="wide")

# --- FUNCIONES DE DATOS Y MODELOS ---

@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    """Carga y prepara los datos iniciales desde el CSV."""
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    """Carga los modelos pre-entrenados y sus métricas desde el archivo joblib."""
    try:
        datos_modelos = joblib.load(file_name)
        # (El mensaje de éxito ya se muestra en la página principal 1_Bienvenida.py)
        # st.sidebar.success(f"Modelos de IA cargados (Entrenados el {datos_modelos['fecha_entrenamiento'].strftime('%d-%m-%Y')})")
        return datos_modelos
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo de modelos '{file_name}'.")
        st.info("Por favor, ejecuta el script 'train_models.py' en tu terminal para generar el archivo.")
        return None

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

# --- ¡NUEVAS FUNCIONES AVANZADAS PARA PREDICCIÓN! ---

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
    Crea las features (lag, rolling, clima, festivos) para el set de predicción,
    usando los datos históricos como base.
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
        df['Temperatura_Media'] = 15.0
    
    # 3. Características de Retraso (Lag) y Móviles (Rolling)
    df = df.set_index('ds') # Volver a poner 'ds' como índice para shift/rolling
    df['ventas_lag_1'] = df['y'].shift(1)
    df['ventas_lag_7'] = df['y'].shift(7)
    df['media_movil_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean()
    df['media_movil_30d'] = df['y'].shift(1).rolling(window=30, min_periods=1).mean()
    
    df = df.bfill().reset_index() # Devolver 'ds' como columna
    
    return df.iloc[-len(df_futuro):]

def generar_pronostico_avanzado(model, df_historico, df_clima, dias_a_pronosticar, fecha_maxima_historica):
    """
    Genera la predicción de demanda usando el modelo AVANZADO.
    """
    fecha_max_dt = pd.to_datetime(fecha_maxima_historica)
    fechas_futuras = pd.date_range(
        start=fecha_max_dt + pd.Timedelta(days=1),
        periods=dias_a_pronosticar, freq='D'
    )
    df_futuro_base = pd.DataFrame({'ds': fechas_futuras, 'y': np.nan})
    
    df_diario = df_historico.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    
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
    
    df_futuro_base['Prediccion'] = prediccion_futura.round(0).astype(int)
    return df_futuro_base # Devolvemos el df completo (con 'ds' como columna)
# --- FIN DE FUNCIONES ---

# --- INICIO DE LA APLICACIÓN ---
st.title("Resumen General y Pronóstico de Demanda")

df_total = cargar_datos()
datos_modelos = cargar_modelos() 
# --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN 2!! ---
df_clima = cargar_clima() # Cargar los datos climáticos

if df_total is not None and datos_modelos is not None:
    
    # --- BARRA LATERAL (EL MENÚ DE INTERACCIÓN) ---
    st.sidebar.title("Menú de Filtros")
    st.sidebar.header("Filtros Globales")
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    
    rango_fechas = st.sidebar.date_input(
        "Selecciona un rango de fechas:",
        value=[fecha_min, fecha_max],
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_seleccionada = st.sidebar.selectbox(
        "Selecciona una Farmacia:",
        options=lista_farmacias
    )
    
    categorias_interes = ['Alergia', 'Antigripal']
    lista_productos_interes = sorted(
        df_total[df_total['Categoria'].isin(categorias_interes)]['Producto'].unique()
    )

    # --- DATOS FILTRADOS ---
    if len(rango_fechas) == 2:
        df_filtrado = df_total[
            (df_total['Fecha'] >= rango_fechas[0]) &
            (df_total['Fecha'] <= rango_fechas[1])
        ]
        if farmacia_seleccionada != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Farmacia_ID'] == farmacia_seleccionada]
    else:
        df_filtrado = df_total.copy()

    # --- INICIO DE LA ESTRUCTURA DE PESTAÑAS ---
    
    tab1, tab2, tab3 = st.tabs([
        "KPIs y Métricas Clave", 
        "Análisis de Épocas", 
        "Pronóstico de IA"
    ])

    # --- PESTAÑA 1: KPIs y Métricas ---
    with tab1:
        st.header(f"Métricas para: {farmacia_seleccionada}")
        total_ventas = df_filtrado['Total_Venta_€'].sum()
        total_unidades = df_filtrado['Cantidad'].sum()
        num_transacciones = len(df_filtrado) 
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas Totales (€)", f"{total_ventas:,.2f} €")
        col2.metric("Unidades Totales Vendidas", f"{total_unidades:,.0f}")
        col3.metric("Transacciones Registradas", f"{num_transacciones:,.0f}")
        st.divider()
        st.subheader("Ventas por Categoría")
        df_cat_agg = df_filtrado.groupby('Categoria')['Total_Venta_€'].sum().reset_index()
        chart_barras = alt.Chart(df_cat_agg).mark_bar().encode(
            x=alt.X('Categoria', title=None), y=alt.Y('Total_Venta_€', title='Ventas Totales (€)'),
            color='Categoria', tooltip=['Categoria', 'Total_Venta_€']
        ).interactive()
        st.altair_chart(chart_barras, use_container_width=True)

    # --- PESTAÑA 2: Análisis de Épocas ---
    with tab2:
        st.header("Análisis Estacional: Alergia vs. Antigripal")
        st.markdown("Agregación semanal de unidades vendidas.")
        df_grafico_epocas = df_filtrado[df_filtrado['Categoria'].isin(categorias_interes)]
        df_grafico_epocas['Fecha'] = pd.to_datetime(df_grafico_epocas['Fecha'])
        if not df_grafico_epocas.empty:
            df_grafico_epocas.set_index('Fecha', inplace=True)
            df_semanal = df_grafico_epocas.groupby('Categoria').resample('W')['Cantidad'].sum().reset_index()
            chart_epocas = alt.Chart(df_semanal).mark_line(point=True).encode(
                x=alt.X('Fecha', title='Fecha (Semanas)'), y=alt.Y('Cantidad', title='Cantidad Total Vendida'),
                color=alt.Color('Categoria', title='Categoría'), tooltip=['Fecha', 'Categoria', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_epocas, use_container_width=True)
        else:
            st.warning("No hay datos para 'Alergia' o 'Antigripal' en los filtros seleccionados.")
        
        st.divider()
        st.header("Análisis de Componentes (Tendencia y Estacionalidad)")
        st.markdown("Descompone la serie temporal de un producto para entender sus patrones subyacentes.")
        st.info("""
        Este análisis separa las ventas de un producto en tres partes:
        - **Tendencia:** La dirección general de las ventas a largo plazo (¿está creciendo o decreciendo?).
        - **Estacionalidad:** El patrón repetitivo que ocurre cada año (ej. el pico de gripe en invierno).
        - **Residuo:** El "ruido" aleatorio que no se puede explicar por las dos anteriores.
        """)
        producto_decomp = st.selectbox("Selecciona un producto para descomponer:", options=lista_productos_interes, index=0)
        with st.expander(f"Ver descomposición para '{producto_decomp}'"):
            try:
                df_producto = df_total[df_total['Producto'] == producto_decomp].copy()
                if farmacia_seleccionada != 'Todas':
                    df_producto = df_producto[df_producto['Farmacia_ID'] == farmacia_seleccionada]
                df_producto['Fecha'] = pd.to_datetime(df_producto['Fecha'])
                df_diario = df_producto.groupby('Fecha')['Cantidad'].sum()
                all_dates = pd.date_range(start=df_diario.index.min(), end=df_diario.index.max(), freq='D')
                df_diario = df_diario.reindex(all_dates, fill_value=0)
                if len(df_diario) < (365 * 2):
                    st.warning(f"No hay suficientes datos (se necesitan 2 años) para '{producto_decomp}' en esta farmacia.")
                else:
                    decomposition = seasonal_decompose(df_diario, model='additive', period=365)
                    df_decomp = pd.DataFrame({'Tendencia': decomposition.trend, 'Estacionalidad': decomposition.seasonal, 'Residuo': decomposition.resid}).reset_index().rename(columns={'index': 'Fecha'})
                    df_decomp_melted = df_decomp.melt('Fecha', var_name='Componente', value_name='Valor')
                    chart_decomp = alt.Chart(df_decomp_melted).mark_line().encode(
                        x=alt.X('Fecha', title=''), y=alt.Y('Valor', title=None), color='Componente', tooltip=['Fecha', 'Componente', 'Valor']
                    ).properties(title=f"Descomposición de '{producto_decomp}' (Diario)").facet(
                        row=alt.Row('Componente', title=None, sort=['Tendencia', 'Estacionalidad', 'Residuo']),
                        resolve=alt.Resolve(scale={'y': 'independent'})
                    ).interactive()
                    st.altair_chart(chart_decomp, use_container_width=True)
            except Exception as e:
                st.error(f"No se pudo descomponer la serie temporal. Causa probable: datos insuficientes.")

    # --- PESTAÑA 3: Pronóstico de IA ---
    with tab3:
        st.header(f"Pronóstico de Demanda para: {farmacia_seleccionada}")
        st.sidebar.divider() 
        st.sidebar.header("Filtros de Pronóstico (IA)")
        producto_pronostico = st.sidebar.selectbox("Selecciona un Producto para Pronosticar:", options=lista_productos_interes, key='pronostico_producto')
        dias_a_pronosticar = st.sidebar.slider("Días a pronosticar:", 30, 120, 90, 15)
        
        if st.button(f"Generar pronóstico de {dias_a_pronosticar} días", type="primary"):
            if datos_modelos is None:
                st.error("No se pueden generar pronósticos porque el archivo de modelos no está cargado.")
            else:
                with st.spinner("Buscando modelo inteligente y generando pronóstico..."):
                    clave_modelo = f"{farmacia_seleccionada}::{producto_pronostico}"
                    modelos_cargados = datos_modelos['modelos']
                    
                    info_modelo_seleccionado = modelos_cargados.get(clave_modelo)
                    
                    if info_modelo_seleccionado is None:
                        st.error(f"No se encontró un modelo pre-entrenado para '{producto_pronostico}' en '{farmacia_seleccionada}'.")
                        st.info("Esto puede ser porque no tenía suficientes datos históricos para un entrenamiento.")
                    else:
                        modelo = info_modelo_seleccionado['model']
                        rmse_modelo = info_modelo_seleccionado['rmse']
                        df_importancia = info_modelo_seleccionado['importance']
                        
                        df_historico_producto = df_total[df_total['Producto'] == producto_pronostico].copy()
                        if farmacia_seleccionada != 'Todas':
                            df_historico_producto = df_historico_producto[df_historico_producto['Farmacia_ID'] == farmacia_seleccionada]
                        
                        df_futuro = generar_pronostico_avanzado(modelo, df_historico_producto, df_clima, dias_a_pronosticar, fecha_max)
                        
                        df_real = df_historico_producto.groupby('Fecha')['Cantidad'].sum().reset_index()
                        df_real = df_real.rename(columns={'Fecha': 'ds', 'Cantidad': 'Ventas'})
                        df_real['Tipo'] = 'Real'
                        df_real['ds'] = pd.to_datetime(df_real['ds'])
                        df_real_reciente = df_real[df_real['ds'] > (pd.to_datetime(fecha_max) - pd.Timedelta(days=365))]
                        
                        df_plot_pred = df_futuro[['ds', 'Prediccion']].rename(columns={'Prediccion': 'Ventas'})
                        df_plot_pred['Tipo'] = 'Predicción'
                        
                        df_plot_combinado = pd.concat([df_real_reciente, df_plot_pred], ignore_index=True)
                        
                        chart_pronostico = alt.Chart(df_plot_combinado).mark_line().encode(
                            x=alt.X('ds', title='Fecha'), y=alt.Y('Ventas', title='Cantidad Vendida'),
                            color=alt.Color('Tipo', title='Dato'), strokeDash=alt.StrokeDash('Tipo', title='Dato'),
                            tooltip=['ds', 'Ventas', 'Tipo']
                        ).interactive()
                        
                        st.altair_chart(chart_pronostico, use_container_width=True)
                        st.success("¡Pronóstico generado con el modelo avanzado!")
                        
                        st.divider()
                        st.header("Análisis del Modelo de IA ('Caja de Cristal')")
                        
                        col_metrica, col_impulsores = st.columns([1, 2])
                        
                        with col_metrica:
                            st.subheader("Fiabilidad")
                            st.metric(
                                label="Error Promedio (RMSE)",
                                value=f"{rmse_modelo:.2f} unidades/día",
                                help="RMSE (Error Cuadrático Medio) indica el error promedio de predicción del modelo. Un valor más bajo es mejor."
                            )
                        
                        with col_impulsores:
                            st.subheader("Impulsores Clave")
                            
                            chart_imp = alt.Chart(df_importancia.head(7)).mark_bar().encode(
                                x=alt.X('Importancia:Q', title='Importancia Relativa'),
                                y=alt.Y('Impulsor:N', title='Factor', sort='-x')
                            ).properties(
                                title="Top 7 Factores para la Predicción"
                            ).interactive()
                            st.altair_chart(chart_imp, use_container_width=True)
                            st.markdown("Este gráfico muestra qué factores considera el modelo más importantes para hacer su predicción.")

                        st.subheader("Datos del Pronóstico")
                        st.dataframe(df_futuro.rename(columns={'ds': 'Fecha', 'Prediccion': 'Cantidad_Pronosticada'}).set_index('Fecha'), use_container_width=True)

else:
    st.error("Error al cargar los datos. Revisa el archivo CSV.")