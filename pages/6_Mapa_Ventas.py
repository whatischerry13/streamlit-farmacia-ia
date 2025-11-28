import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk

# --- 1. Configuración de Página (con layout="wide") ---
st.set_page_config(page_title="Mapa de Ventas", layout="wide")

# --- 2. Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- Funciones de Datos (CON CACHÉ) ---
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
def simular_coordenadas(_df_total):
    """
    Simula Latitud y Longitud para cada farmacia única
    basándose en su 'Zona_Farmacia'.
    """
    farmacias = _df_total[['Farmacia_ID', 'Zona_Farmacia']].drop_duplicates()
    coordenadas_base = {'Centro': [40.4168, -3.7038], 'Norte': [40.4531, -3.6883], 'Sur': [40.3833, -3.7167]}
    np.random.seed(42)
    latitudes = []
    longitudes = []
    for zona in farmacias['Zona_Farmacia']:
        base_lat, base_lon = coordenadas_base.get(zona, [40.41, -3.70])
        latitudes.append(base_lat + np.random.uniform(-0.01, 0.01))
        longitudes.append(base_lon + np.random.uniform(-0.01, 0.01))
    farmacias['lat'] = latitudes
    farmacias['lon'] = longitudes
    return farmacias

# --- INTERFAZ DE STREAMLIT ---
st.title("Mapa de Rendimiento Geoespacial")

# --- 3. Explicación Profesional ---
st.info("""
**¿Para qué sirve esto?**
Esta sección proporciona un análisis geoespacial 3D (usando Pydeck) del rendimiento de las farmacias.
Permite a los gerentes de zona identificar visualmente las "zonas calientes" de ventas o unidades vendidas, filtrando por categoría y rango de fechas.
""")

df_total = cargar_datos()
if df_total is not None:
    df_coordenadas = simular_coordenadas(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Filtros del Mapa")
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    cat_sel = st.sidebar.selectbox("Filtrar por Categoría:", options=lista_categorias, index=0, key='mapa_categoria')
    
    # Radio con texto limpio
    metrica_sel = st.sidebar.radio(
        "Métrica a Visualizar:",
        options=['Total Ventas (Euros)', 'Total Unidades'], 
        key='mapa_metrica'
    )
    
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Selecciona un rango de fechas:", value=[fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max, key='mapa_fechas')

    # --- Aplicar Filtros ---
    if len(rango_fechas) == 2:
        df_filtrado = df_total[(df_total['Fecha'] >= rango_fechas[0]) & (df_total['Fecha'] <= rango_fechas[1])]
        if cat_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Categoria'] == cat_sel]
    else:
        df_filtrado = df_total.copy()

    # --- AGREGACIÓN DE DATOS PARA EL MAPA ---
    df_agg_farmacia = df_filtrado.groupby('Farmacia_ID').agg(
        Total_Ventas_Euros=('Total_Venta_€', 'sum'),
        Total_Unidades=('Cantidad', 'sum')
    ).reset_index()
    
    df_mapa = df_coordenadas.merge(df_agg_farmacia, on='Farmacia_ID', how='left').fillna(0)
    df_mapa['Ventas_str'] = df_mapa['Total_Ventas_Euros'].apply(lambda x: f"{x:,.0f} €")
    df_mapa['Unidades_str'] = df_mapa['Total_Unidades'].apply(lambda x: f"{x:,.0f}")
    
    # Paleta de colores profesional y suave
    zona_color_map = {
        'Centro': [0, 128, 255, 180], # Azul (Azure)
        'Norte': [34, 139, 34, 180],  # Verde (ForestGreen)
        'Sur': [205, 92, 92, 180]     # Rojo Suave (IndianRed)
    }
    df_mapa['color'] = df_mapa['Zona_Farmacia'].apply(lambda z: zona_color_map.get(z, [128, 128, 128, 160]))

    if metrica_sel == 'Total Ventas (Euros)':
        columna_size = 'Total_Ventas_Euros'
    else:
        columna_size = 'Total_Unidades'
        
    max_val = df_mapa[columna_size].max()
    if max_val > 0:
        df_mapa['radius_meters'] = (df_mapa[columna_size] / max_val) * 500 + 100
    else:
        df_mapa['radius_meters'] = 100

    st.header(f"Mostrando: {metrica_sel} para '{cat_sel}'")

    if df_mapa.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        # --- Creación del Mapa Pydeck ---
        try:
            view_state = pdk.ViewState(
                latitude=df_mapa['lat'].mean(),
                longitude=df_mapa['lon'].mean(),
                zoom=11.5,
                pitch=50 
            )
            scatterplot_layer = pdk.Layer(
                'ScatterplotLayer', data=df_mapa, get_position='[lon, lat]', get_color='color',
                get_radius='radius_meters', opacity=0.7, stroked=True, filled=True,
                wireframe=True, get_line_color=[255, 255, 255, 100], get_line_width_min_pixels=1,
                pickable=True, auto_highlight=True, highlight_color=[255, 255, 255, 200]
            )
            text_layer = pdk.Layer(
                'TextLayer', data=df_mapa, get_position='[lon, lat]', get_text='Farmacia_ID',
                get_size=12, get_alignment_baseline="'bottom'", get_text_anchor="'middle'",
                get_pixel_offset=[0, -15], get_color=[255, 255, 255, 200], 
                outline_width=2, outline_color=[0, 0, 0, 255], pickable=False
            )
            tooltip = {
                "html": """<div style="background:#222;color:white;padding:10px;border-radius:5px;font-family:sans-serif;">
                           <b>{Farmacia_ID}</b> ({Zona_Farmacia})<br/>
                           <b>Ventas:</b> {Ventas_str}<br/>
                           <b>Unidades:</b> {Unidades_str}</div>""",
                "style": {"backgroundColor": "rgba(0,0,0,0)", "border": "none"}
            }

            # Esta es la llamada delicada. La dejamos exactamente como estaba
            # cuando funcionó (sin el mapbox_key explícito en pdk.Deck)
            r = pdk.Deck(
                layers=[scatterplot_layer, text_layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/dark-v9',
                tooltip=tooltip
            )
            st.pydeck_chart(r, use_container_width=True)

            st.subheader("Datos Detallados del Mapa")
            
            # --- 4. Botón de Descarga ---
            df_display_mapa = df_mapa[['Farmacia_ID', 'Zona_Farmacia', 'Total_Ventas_Euros', 'Total_Unidades', 'lat', 'lon']]
            st.dataframe(df_display_mapa, use_container_width=True)
            
            csv_data = convert_df_to_csv(df_display_mapa)
            st.download_button(
                label="Descargar Datos del Mapa en CSV",
                data=csv_data,
                file_name=f"reporte_mapa_ventas.csv",
                mime='text/csv',
                use_container_width=True
            )

        # Mantenemos los bloques try/except por seguridad
        except KeyError as e:
            if "MAPBOX_API_KEY" in str(e):
                 st.error("Error de Clave Mapbox: No se encontró 'MAPBOX_API_KEY' en los Secrets.")
                 st.info("Asegúrate de haber añadido tu token de Mapbox a los Secrets de Streamlit Cloud (Manage app -> Settings -> Secrets) con el formato correcto: MAPBOX_API_KEY = \"pk.ey...\".")
            else:
                 st.error(f"Error inesperado al crear el mapa: Clave no encontrada '{e}'")
        except Exception as e:
            st.error(f"Error al renderizar el mapa Pydeck: {e}")

else:
    st.error("Error al cargar los datos.")