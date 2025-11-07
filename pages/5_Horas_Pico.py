import pandas as pd
import streamlit as st
import numpy as np
import altair as alt

# --- 1. Configuración de Página ---
st.set_page_config(page_title="Análisis Horario", layout="wide")

# --- 2. Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- Funciones de Datos ---
@st.cache_data
def cargar_y_simular_timestamp(file_name='ventas_farmacia_fake.csv'):
    """
    Carga los datos y simula la columna 'Timestamp' (con hora).
    ¡AHORA TAMBIÉN CREA EL DÍA DE LA SEMANA!
    """
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        
        # --- Simulación de Hora ---
        df['Timestamp'] = pd.to_datetime(df['Fecha'])
        np.random.seed(42)
        segundos_aleatorios = np.random.randint(28800, 75600, size=len(df)) # 8:00 a 21:00
        df['Timestamp'] = df['Timestamp'] + pd.to_timedelta(segundos_aleatorios, unit='s')
        
        # --- MEJORA: Extraer Hora y Día de la Semana ---
        df['Hora_del_Dia'] = df['Timestamp'].dt.hour
        df['Dia_Semana_Num'] = df['Timestamp'].dt.dayofweek # 0=Lunes, 6=Domingo
        dias_mapa = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        df['Dia_Semana'] = df['Dia_Semana_Num'].map(dias_mapa)
        
        df['Fecha'] = df['Timestamp'].dt.date
        return df
    
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        return None

# --- INTERFAZ DE STREAMLIT ---

st.title("Análisis de Demanda Horaria (Matriz de Demanda)")
st.info("""
**¿Para qué sirve esto?**
Esta página analiza *cuándo* se producen las ventas. El gráfico principal es un mapa de calor que cruza el **Día de la Semana** con la **Hora del Día** para encontrar las "zonas calientes" de actividad en la farmacia.
""", icon="ℹ️")

st.warning("""
**Nota sobre la Calidad de los Datos:**
El dataset original **no contiene la hora** de la transacción. Las horas han sido **simuladas** aleatoriamente entre las 8:00 y las 21:00 para demostrar la funcionalidad. 
En un proyecto real, los datos del TPV (Terminal Punto de Venta) mostrarían los patrones reales.
""")

df_total = cargar_y_simular_timestamp()

if df_total is not None:

    # --- Filtros de la Barra Lateral ---
    st.sidebar.title("Filtros de Análisis")
    
    # --- MEJORA: Radio para seleccionar la métrica ---
    metrica_sel = st.sidebar.radio(
        "Selecciona la Métrica a Visualizar:",
        options=['Total Unidades', 'Total Ventas (€)'],
        key='horas_pico_metrica',
        horizontal=True
    )
    
    st.sidebar.divider()
    
    lista_farmacias = ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Filtrar por Farmacia:", options=lista_farmacias, key='horas_pico_farmacia')
    
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    try: default_index = lista_categorias.index('Antigripal')
    except ValueError: default_index = 0
    cat_sel = st.sidebar.selectbox("Filtrar por Categoría:", options=lista_categorias, index=default_index, key='horas_pico_categoria')
    
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Selecciona un rango de fechas:", value=[fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max, key='horas_pico_fechas')
    
    # --- Aplicar Filtros ---
    if len(rango_fechas) == 2:
        df_filtrado = df_total[(df_total['Fecha'] >= rango_fechas[0]) & (df_total['Fecha'] <= rango_fechas[1])]
        if farmacia_sel != 'Todas': df_filtrado = df_filtrado[df_filtrado['Farmacia_ID'] == farmacia_sel]
        if cat_sel != 'Todas': df_filtrado = df_filtrado[df_filtrado['Categoria'] == cat_sel]
    else:
        df_filtrado = df_total.copy()

    
    # --- ANÁLISIS Y GRÁFICO DE MAPA DE CALOR ---
    
    if metrica_sel == 'Total Unidades':
        metrica_a_usar = 'Cantidad'
    else:
        metrica_a_usar = 'Total_Venta_€'
        
    st.header(f"Mapa de Calor: {metrica_sel} por Día y Hora")
    st.markdown(f"Análisis para: **{farmacia_sel}** | Categoría: **{cat_sel}**")
    
    if df_filtrado.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        # 1. Agrupar datos para el heatmap
        df_heatmap = df_filtrado.groupby(['Dia_Semana', 'Hora_del_Dia'])[metrica_a_usar].sum().reset_index()
        
        # 2. Definir el orden correcto de los días de la semana (para que no se ordenen alfabéticamente)
        orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        # 3. Crear el gráfico de Altair
        heatmap = alt.Chart(df_heatmap).mark_rect().encode(
            # Eje X: Hora del día. ':O' significa Ordinal (categórico ordenado)
            x=alt.X('Hora_del_Dia:O', title="Hora del Día"),
            
            # Eje Y: Día de la semana, con el orden personalizado
            y=alt.Y('Dia_Semana:O', title="Día de la Semana", sort=orden_dias),
            
            # Color: La intensidad de la métrica
            color=alt.Color(metrica_a_usar, 
                            title=metrica_sel,
                            scale=alt.Scale(range='heatmap') # Esquema de color predefinido
                           ),
            
            # Tooltip: Información al pasar el ratón
            tooltip=['Dia_Semana', 'Hora_del_Dia', alt.Tooltip(metrica_a_usar, title=metrica_sel, format=',.0f')]
        ).properties(
            title=f"Patrón de Demanda Semanal ({metrica_sel})"
        ).interactive() # Permite zoom y pan
        
        st.altair_chart(heatmap, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Datos del Mapa de Calor")
        st.dataframe(df_heatmap, use_container_width=True)
        
        # 4. Botón de Descarga
        csv_data = convert_df_to_csv(df_heatmap)
        st.download_button(
            label=f"Descargar Datos del Mapa de Calor en CSV",
            data=csv_data,
            file_name=f"reporte_horas_pico.csv",
            mime='text/csv',
            use_container_width=True
        )

else:
    st.error("Error al cargar los datos.")