import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st # <-- ¡Nueva importación!
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURACIÓN DE LA PÁGINA ---
# Esto debe ser lo primero que ejecuta Streamlit
st.set_page_config(
    page_title="Dashboard de Ventas Farmacia",
    page_icon="⚕️",
    layout="wide" # Ocupa toda la pantalla
)

# --- CACHING DE DATOS ---
# @st.cache_data le dice a Streamlit: "No cargues este CSV cada vez
# que el usuario mueva un slider. Cárgalo UNA VEZ y guárdalo en memoria."
# Esto es VITAL para la velocidad de la app.
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
        # Aseguramos que 'Fecha' no tenga hora (solo fecha)
        df['Fecha'] = df['Fecha'].dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

# @st.cache_resource le dice a Streamlit: "Entrenar este modelo de IA es
# costoso. Entrénalo UNA VEZ por cada set de parámetros y guarda el modelo."
@st.cache_resource
def entrenar_modelo_pronostico(_df, producto, farmacia_id):
    """
    Entrena un modelo XGBoost para un producto y farmacia específicos.
    El argumento _df (con '_') le indica a cache_resource que no "mire"
    dentro del dataframe, solo que sepa que existe.
    """
    
    # 1. Preparar datos
    df_producto = _df[_df['Producto'] == producto].copy()
    
    # Si se selecciona una farmacia, filtrar por ella
    if farmacia_id != 'Todas':
        df_producto = df_producto[df_producto['Farmacia_ID'] == farmacia_id]
        
    df_diario = df_producto.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    
    # Convertir 'ds' (que es date) a datetime para las features
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])

    # 2. Ingeniería de Características (Función anidada)
    def crear_caracteristicas_temporales(df_in):
        df_out = df_in.copy()
        df_out['mes'] = df_out['ds'].dt.month
        df_out['dia_del_ano'] = df_out['ds'].dt.dayofyear
        df_out['semana_del_ano'] = df_out['ds'].dt.isocalendar().week.astype(int)
        df_out['dia_de_la_semana'] = df_out['ds'].dt.dayofweek
        df_out['ano'] = df_out['ds'].dt.year
        df_out['trimestre'] = df_out['ds'].dt.quarter
        return df_out

    df_preparado = crear_caracteristicas_temporales(df_diario)
    
    # Si no hay datos (ej. farmacia nueva), retornar None
    if df_preparado.empty:
        return None

    # 3. Definir Características (X) y Objetivo (y)
    features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
    target = 'y'

    X = df_preparado[features]
    y = df_preparado[target]

    # 4. Entrenar el Modelo XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )

    eval_size = int(len(X) * 0.2)
    # Manejar el caso de muy pocos datos
    if eval_size < 50: 
        model.fit(X, y, verbose=False)
    else:
        X_train, X_eval = X.iloc[:-eval_size], X.iloc[-eval_size:]
        y_train, y_eval = y.iloc[:-eval_size], y.iloc[-eval_size:]
        model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False
        )
    
    return model

# --- INICIO DE LA APLICACIÓN ---

# Título del Dashboard
st.title("⚕️ Dashboard de Análisis y Pronóstico para Farmacias")

# Cargar los datos (usará el caché)
df_total = cargar_datos()

if df_total is not None:
    
    # --- BARRA LATERAL (EL MENÚ DE INTERACCIÓN) ---
    st.sidebar.title("Menú de Filtros")
    
    # Filtro 1: Rango de Fechas (Marketing/Ventas)
    st.sidebar.header("Filtros de Análisis (Ventas)")
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    
    rango_fechas = st.sidebar.date_input(
        "Selecciona un rango de fechas:",
        value=[fecha_min, fecha_max], # Valor por defecto
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    # Filtro 2: Farmacia (Marketing/Ventas)
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_seleccionada = st.sidebar.selectbox(
        "Selecciona una Farmacia:",
        options=lista_farmacias
    )

    # --- DATOS FILTRADOS ---
    # Creamos un dataframe filtrado basado en la selección
    if len(rango_fechas) == 2:
        df_filtrado = df_total[
            (df_total['Fecha'] >= rango_fechas[0]) &
            (df_total['Fecha'] <= rango_fechas[1])
        ]
        
        if farmacia_seleccionada != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Farmacia_ID'] == farmacia_seleccionada]
    else:
        # Si el usuario no ha puesto un rango válido, usamos todo
        df_filtrado = df_total.copy()


    # --- SECCIÓN 1: ANÁLISIS DE VENTAS (MARKETING) ---
    st.header(f"Análisis de Ventas para: {farmacia_seleccionada}")
    
    # Métricas clave (KPIs)
    total_ventas = df_filtrado['Total_Venta_€'].sum()
    total_unidades = df_filtrado['Cantidad'].sum()
    
    col1, col2 = st.columns(2) # Crear dos columnas
    col1.metric("Ventas Totales (€)", f"{total_ventas:,.2f} €")
    col2.metric("Unidades Totales Vendidas", f"{total_unidades:,.0f}")
    
    # Gráfico de Épocas (el que hicimos antes, pero ahora es interactivo)
    st.subheader("Ventas por Categoría (Alergia vs. Antigripal)")
    
    categorias_interes = ['Alergia', 'Antigripal']
    df_grafico_epocas = df_filtrado[df_filtrado['Categoria'].isin(categorias_interes)]
    
    # Convertir 'Fecha' (date) a datetime para resamplear
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
        
        # st.altair_chart lo dibuja en la app
        st.altair_chart(chart_epocas, use_container_width=True)
    else:
        st.warning("No hay datos para las categorías 'Alergia' o 'Antigripal' en los filtros seleccionados.")

    
    # --- SECCIÓN 2: PRONÓSTICO DE IA (VENTAS/STOCK) ---
    st.sidebar.divider() # Separador en la barra lateral
    st.sidebar.header("Filtros de Pronóstico (IA)")
    
    # Filtro 3: Producto a Pronosticar
    lista_productos = sorted(
        df_total[df_total['Categoria'].isin(categorias_interes)]['Producto'].unique()
    )
    producto_pronostico = st.sidebar.selectbox(
        "Selecciona un Producto para Pronosticar:",
        options=lista_productos
    )
    
    dias_a_pronosticar = st.sidebar.slider(
        "Días a pronosticar:",
        min_value=30, max_value=120, value=90, step=15
    )
    
    st.header(f"Pronóstico de Demanda para: {producto_pronostico}")
    st.subheader(f"(Filtrado por Farmacia: {farmacia_seleccionada})")

    # Botón para ejecutar el pronóstico
    if st.button(f"Generar pronóstico de {dias_a_pronosticar} días"):
        
        # Usamos un 'spinner' para que el usuario sepa que está trabajando
        with st.spinner("Entrenando modelo de IA y generando pronóstico..."):
            
            # 1. Entrenar (usará el caché si ya lo hizo)
            model = entrenar_modelo_pronostico(df_total, producto_pronostico, farmacia_seleccionada)
            
            if model is None:
                st.error("No hay suficientes datos históricos para este producto y farmacia para entrenar un modelo.")
            else:
                # 2. Generar fechas futuras y predicción
                
                # Función anidada para crear features (debe ser idéntica a la de entrenamiento)
                def crear_caracteristicas_temporales(df_in):
                    df_out = df_in.copy()
                    df_out['mes'] = df_out['ds'].dt.month
                    df_out['dia_del_ano'] = df_out['ds'].dt.dayofyear
                    df_out['semana_del_ano'] = df_out['ds'].dt.isocalendar().week.astype(int)
                    df_out['dia_de_la_semana'] = df_out['ds'].dt.dayofweek
                    df_out['ano'] = df_out['ds'].dt.year
                    df_out['trimestre'] = df_out['ds'].dt.quarter
                    return df_out

                fecha_maxima_historica = pd.to_datetime(df_total['Fecha'].max())
                fechas_futuras = pd.date_range(
                    start=fecha_maxima_historica + pd.Timedelta(days=1),
                    periods=dias_a_pronosticar,
                    freq='D'
                )
                
                df_futuro = pd.DataFrame({'ds': fechas_futuras})
                df_futuro_preparado = crear_caracteristicas_temporales(df_futuro)
                
                features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
                prediccion_futura = model.predict(df_futuro_preparado[features])
                prediccion_futura[prediccion_futura < 0] = 0
                df_futuro['Prediccion'] = prediccion_futura.round(0).astype(int)

                # 3. Preparar datos para el gráfico final
                
                # Datos reales
                df_real = df_filtrado[df_filtrado['Producto'] == producto_pronostico].copy()
                df_real = df_real.groupby('Fecha')['Cantidad'].sum().reset_index()
                df_real = df_real.rename(columns={'Fecha': 'ds', 'Cantidad': 'Ventas'})
                df_real['Tipo'] = 'Real'
                df_real['ds'] = pd.to_datetime(df_real['ds']) # Asegurar datetime
                
                # Solo último año de datos reales
                df_real_reciente = df_real[df_real['ds'] > (fecha_maxima_historica - pd.Timedelta(days=365))]
                
                # Datos predichos
                df_plot_pred = df_futuro[['ds', 'Prediccion']].rename(columns={'Prediccion': 'Ventas'})
                df_plot_pred['Tipo'] = 'Predicción'
                
                df_plot_combinado = pd.concat([df_real_reciente, df_plot_pred])
                
                # 4. Dibujar el gráfico
                chart_pronostico = alt.Chart(df_plot_combinado).mark_line().encode(
                    x=alt.X('ds', title='Fecha'),
                    y=alt.Y('Ventas', title='Cantidad Vendida'),
                    color=alt.Color('Tipo', title='Dato'),
                    strokeDash=alt.StrokeDash('Tipo', title='Dato'),
                    tooltip=['ds', 'Ventas', 'Tipo']
                ).interactive()
                
                st.altair_chart(chart_pronostico, use_container_width=True)
                
                st.success("¡Pronóstico generado!")
                
                # Mostrar los datos del pronóstico en una tabla
                st.subheader("Datos del Pronóstico")
                st.dataframe(df_futuro.rename(columns={'ds': 'Fecha', 'Prediccion': 'Cantidad_Pronosticada'}).set_index('Fecha'))

else:
    st.error("Error al cargar los datos. Revisa el archivo CSV.")