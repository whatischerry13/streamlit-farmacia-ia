import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Análisis de Horas Pico", page_icon="⏰", layout="wide")

# --- FUNCIONES DE DATOS (CON CACHÉ) ---

@st.cache_data
def cargar_y_simular_timestamp(file_name='ventas_farmacia_fake.csv'):
    """
    Carga los datos y simula la columna 'Timestamp' (con hora)
    ya que el CSV original no la tiene.
    """
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        
        # --- Simulación de Hora ---
        # Convertimos la 'Fecha' (que es medianoche) a datetime
        df['Timestamp'] = pd.to_datetime(df['Fecha'])
        
        # Generamos horas aleatorias (en segundos) para cada venta
        # Simulamos un horario de farmacia (de 8:00 a 21:00)
        # 8:00 = 8 * 3600 = 28800 segundos
        # 21:00 = 21 * 3600 = 75600 segundos
        np.random.seed(42) # Para reproducibilidad
        segundos_aleatorios = np.random.randint(
            28800, 
            75600, 
            size=len(df)
        )
        
        # Sumamos los segundos aleatorios a la fecha (que era medianoche)
        df['Timestamp'] = df['Timestamp'] + pd.to_timedelta(segundos_aleatorios, unit='s')
        
        # Extraemos la hora para el análisis
        df['Hora_del_Dia'] = df['Timestamp'].dt.hour
        df['Fecha'] = df['Timestamp'].dt.date
        
        return df
    
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

# --- INTERFAZ DE STREAMLIT ---

st.title("⏰ Análisis de Horas Pico")
st.markdown("Esta sección responde a: *¿A qué hora del día se venden más nuestros productos clave?*")

df_total = cargar_y_simular_timestamp()

if df_total is not None:

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Filtros de Análisis")
    
    # Filtro de Farmacia
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox(
        "Filtrar por Farmacia:",
        options=lista_farmacias,
        key='horas_pico_farmacia'
    )
    
    # Filtro de Categoría (BLOQUE CORREGIDO)
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Buscamos la *posición* (el índice) de 'Antigripal' en la lista
    try:
        default_index = lista_categorias.index('Antigripal')
    except ValueError:
        default_index = 0 # Si no lo encuentra, usa el índice 0 ('Todas')

    cat_sel = st.sidebar.selectbox(
        "Filtrar por Categoría:",
        options=lista_categorias,
        index=default_index, # <-- Usamos 'index' con la posición
        key='horas_pico_categoria'
    )
    # --- FIN DE LA CORRECCIÓN ---
    
    # Filtro de Rango de Fechas
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    
    rango_fechas = st.sidebar.date_input(
        "Selecciona un rango de fechas:",
        value=[fecha_min, fecha_max],
        min_value=fecha_min,
        max_value=fecha_max,
        key='horas_pico_fechas'
    )
    
    # --- Aplicar Filtros ---
    if len(rango_fechas) == 2:
        df_filtrado = df_total[
            (df_total['Fecha'] >= rango_fechas[0]) &
            (df_total['Fecha'] <= rango_fechas[1])
        ]
        
        if farmacia_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Farmacia_ID'] == farmacia_sel]
        
        if cat_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Categoria'] == cat_sel]
    else:
        df_filtrado = df_total.copy()

    
    # --- ANÁLISIS Y GRÁFICO ---
    st.header(f"Patrón de Ventas Horario para: {cat_sel}")
    
    if df_filtrado.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        # Agrupar por la hora del día y sumar la cantidad
        df_agg_horas = df_filtrado.groupby('Hora_del_Dia')['Cantidad'].sum().reset_index()
        
        # Asegurarnos de que todas las horas (8-21) estén presentes, incluso si tienen 0 ventas
        horas_farmacia = pd.DataFrame({'Hora_del_Dia': range(8, 22)})
        df_agg_horas = horas_farmacia.merge(df_agg_horas, on='Hora_del_Dia', how='left').fillna(0)
        
        df_agg_horas = df_agg_horas.set_index('Hora_del_Dia')

        st.markdown(f"Total de unidades vendidas ({cat_sel}) en el período:")
        
        # st.bar_chart es la forma más simple de dibujar un gráfico de barras
        st.bar_chart(df_agg_horas['Cantidad'])
        
        st.markdown(f"""
        **Análisis (simulado):**
        * Este gráfico muestra el total de unidades de `{cat_sel}` vendidas por cada hora del día.
        * Permite al personal de marketing y operaciones optimizar los turnos de personal o lanzar promociones "happy hour".
        """)

else:
    st.error("Error al cargar los datos.")