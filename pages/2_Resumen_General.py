import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
import warnings
import joblib
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose # <-- ¡Nueva importación!

# Ignorar advertencias futuras
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=st.errors.StreamlitAPIWarning)


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
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date # Asegurar formato de fecha
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    """Carga los modelos pre-entrenados desde el archivo joblib."""
    try:
        datos_modelos = joblib.load(file_name)
        # st.success(f"Modelos de IA cargados (entrenados el {datos_modelos['fecha_entrenamiento'].strftime('%Y-%m-%d %H:%M')})")
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
    
    df_futuro['Prediccion'] = prediccion_futura.round(0).astype(int)
    return df_futuro

# --- INICIO DE LA APLICACIÓN ---

st.title("📈 Resumen General y Pronóstico de Demanda")

df_total = cargar_datos()
datos_modelos = cargar_modelos() 

if df_total is not None:
    
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
    
    # --- Definir listas de productos aquí para que ambas pestañas las usen ---
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
        "📊 KPIs y Métricas Clave", 
        "🗓️ Análisis de Épocas", 
        "🤖 Pronóstico de IA"
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
            x=alt.X('Categoria', title=None),
            y=alt.Y('Total_Venta_€', title='Ventas Totales (€)'),
            color='Categoria',
            tooltip=['Categoria', 'Total_Venta_€']
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
            
            df_semanal = df_grafico_epocas.groupby('Categoria').resample('W')['Cantidad'].sum()
            df_semanal = df_semanal.reset_index()

            chart_epocas = alt.Chart(df_semanal).mark_line(point=True).encode(
                x=alt.X('Fecha', title='Fecha (Semanas)'),
                y=alt.Y('Cantidad', title='Cantidad Total Vendida'),
                color=alt.Color('Categoria', title='Categoría'),
                tooltip=['Fecha', 'Categoria', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_epocas, use_container_width=True)
        else:
            st.warning("No hay datos para 'Alergia' o 'Antigripal' en los filtros seleccionados.")

        # --- INICIO DEL NUEVO CÓDIGO (PASO 2) ---
        
        st.divider()
        st.header("Análisis de Componentes (Tendencia y Estacionalidad)")
        st.markdown("Descompone la serie temporal de un producto para entender sus patrones subyacentes.")

        # Selector para el producto a descomponer
        producto_decomp = st.selectbox(
            "Selecciona un producto para descomponer:",
            options=lista_productos_interes,
            index=0 # Por defecto el primero
        )
        
        with st.expander(f"Ver descomposición para '{producto_decomp}'"):
            try:
                # 1. Preparar datos: Usar TODOS los datos (df_total) para un buen análisis
                # y filtrar por el producto y la farmacia seleccionados
                df_producto = df_total[df_total['Producto'] == producto_decomp].copy()
                if farmacia_seleccionada != 'Todas':
                    df_producto = df_producto[df_producto['Farmacia_ID'] == farmacia_seleccionada]
                
                df_producto['Fecha'] = pd.to_datetime(df_producto['Fecha'])
                
                # 2. Agregar diariamente
                df_diario = df_producto.groupby('Fecha')['Cantidad'].sum()
                
                # 3. Rellenar fechas faltantes (CRÍTICO para statsmodels)
                all_dates = pd.date_range(start=df_diario.index.min(), end=df_diario.index.max(), freq='D')
                df_diario = df_diario.reindex(all_dates, fill_value=0)
                
                # 4. Asegurar suficientes datos para 1 ciclo anual (period=365)
                if len(df_diario) < (365 * 2):
                    st.warning(f"No hay suficientes datos (se necesitan 2 años) para '{producto_decomp}' en esta farmacia.")
                else:
                    # 5. Ejecutar la descomposición
                    decomposition = seasonal_decompose(df_diario, model='additive', period=365)
                    
                    # 6. Preparar para graficar con Altair
                    df_decomp = pd.DataFrame({
                        'Tendencia': decomposition.trend,
                        'Estacionalidad': decomposition.seasonal,
                        'Residuo': decomposition.resid
                    }).reset_index().rename(columns={'index': 'Fecha'})
                    
                    df_decomp_melted = df_decomp.melt('Fecha', var_name='Componente', value_name='Valor')
                    
                    # 7. Graficar
                    chart_decomp = alt.Chart(df_decomp_melted).mark_line().encode(
                        x=alt.X('Fecha', title=''),
                        y=alt.Y('Valor', title=None),
                        color='Componente',
                        tooltip=['Fecha', 'Componente', 'Valor']
                    ).properties(
                        title=f"Descomposición de '{producto_decomp}' (Diario)"
                    ).facet(
                        row=alt.Row('Componente', title=None, sort=['Tendencia', 'Estacionalidad', 'Residuo']),
                        resolve=alt.Resolve(scale={'y': 'independent'}) # Ejes Y independientes
                    ).interactive()
                    
                    st.altair_chart(chart_decomp, use_container_width=True)
                    st.markdown("""
                    * **Tendencia:** El patrón a largo plazo (¿están subiendo o bajando las ventas?).
                    * **Estacionalidad:** El patrón repetitivo anual (el pico de gripe/alergia).
                    * **Residuo:** El "ruido" aleatorio que queda.
                    """)
            
            except Exception as e:
                st.error(f"No se pudo descomponer la serie temporal. Causa probable: datos insuficientes o no varianza.")
                st.error(e)
        
        # --- FIN DEL NUEVO CÓDIGO ---


    # --- PESTAÑA 3: Pronóstico de IA ---
    with tab3:
        st.header(f"Pronóstico de Demanda para: {farmacia_seleccionada}")
        
        st.sidebar.divider() 
        st.sidebar.header("Filtros de Pronóstico (IA)")
        
        # Usamos la lista de productos ya definida
        producto_pronostico = st.sidebar.selectbox(
            "Selecciona un Producto para Pronosticar:",
            options=lista_productos_interes,
            key='pronostico_producto' # Añadimos una key única
        )
        
        dias_a_pronosticar = st.sidebar.slider(
            "Días a pronosticar:",
            min_value=30, max_value=120, value=90, step=15
        )
        
        if st.button(f"Generar pronóstico de {dias_a_pronosticar} días", type="primary"):
            
            if datos_modelos is None:
                st.error("No se pueden generar pronósticos porque el archivo de modelos no está cargado.")
            else:
                with st.spinner("Buscando modelo y generando pronóstico... ¡Esto será rápido!"):
                    
                    clave_modelo = f"{farmacia_seleccionada}::{producto_pronostico}"
                    modelos_cargados = datos_modelos['modelos']
                    modelo_seleccionado = modelos_cargados.get(clave_modelo)
                    
                    if modelo_seleccionado is None:
                        st.error(f"No se encontró un modelo pre-entrenado para '{producto_pronostico}' en '{farmacia_seleccionada}'.")
                        st.info("Esto puede ser porque no tenía suficientes datos históricos para un entrenamiento.")
                    else:
                        df_futuro = generar_pronostico(modelo_seleccionado, dias_a_pronosticar, fecha_max)
                        
                        df_real = df_total[df_total['Producto'] == producto_pronostico].copy()
                        if farmacia_seleccionada != 'Todas':
                            df_real = df_real[df_real['Farmacia_ID'] == farmacia_seleccionada]
                        
                        df_real = df_real.groupby('Fecha')['Cantidad'].sum().reset_index()
                        df_real = df_real.rename(columns={'Fecha': 'ds', 'Cantidad': 'Ventas'})
                        df_real['Tipo'] = 'Real'
                        df_real['ds'] = pd.to_datetime(df_real['ds'])
                        
                        df_real_reciente = df_real[df_real['ds'] > (pd.to_datetime(fecha_max) - pd.Timedelta(days=365))]
                        
                        df_plot_pred = df_futuro[['ds', 'Prediccion']].rename(columns={'Prediccion': 'Ventas'})
                        df_plot_pred['Tipo'] = 'Predicción'
                        
                        df_plot_combinado = pd.concat([df_real_reciente, df_plot_pred])
                        
                        chart_pronostico = alt.Chart(df_plot_combinado).mark_line().encode(
                            x=alt.X('ds', title='Fecha'),
                            y=alt.Y('Ventas', title='Cantidad Vendida'),
                            color=alt.Color('Tipo', title='Dato'),
                            strokeDash=alt.StrokeDash('Tipo', title='Dato'),
                            tooltip=['ds', 'Ventas', 'Tipo']
                        ).interactive()
                        
                        st.altair_chart(chart_pronostico, use_container_width=True)
                        st.success("¡Pronóstico generado al instante!")
                        st.dataframe(df_futuro.rename(columns={'ds': 'Fecha', 'Prediccion': 'Cantidad_Pronosticada'}).set_index('Fecha'))

else:
    st.error("Error al cargar los datos. Revisa el archivo CSV.")