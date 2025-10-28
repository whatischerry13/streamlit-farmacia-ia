import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk

st.set_page_config(page_title="Mapa de Ventas", page_icon="🗺️", layout="wide")

# --- FUNCIONES DE DATOS (CON CACHÉ) ---
# ... (tus funciones cargar_datos y simular_coordenadas van aquí, sin cambios) ...
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
def simular_coordenadas(_df_total):
    farmacias = _df_total[['Farmacia_ID', 'Zona_Farmacia']].drop_duplicates()
    coordenadas_base = {'Centro': [40.4168, -3.7038], 'Norte': [40.4531, -3.6883], 'Sur': [40.3833, -3.7167]}
    np.random.seed(42)
    latitudes = []
    longitudes = []
    for zona in farmacias['Zona_Farmacia']:
        base_lat = coordenadas_base.get(zona, [40.41, -3.70])[0]
        base_lon = coordenadas_base.get(zona, [40.41, -3.70])[1]
        latitudes.append(base_lat + np.random.uniform(-0.01, 0.01))
        longitudes.append(base_lon + np.random.uniform(-0.01, 0.01))
    farmacias['lat'] = latitudes
    farmacias['lon'] = longitudes
    return farmacias

# --- INTERFAZ DE STREAMLIT ---
st.title("🗺️ Mapa de Rendimiento Geoespacial")
st.markdown("Análisis 3D de las ventas y rentabilidad por ubicación geográfica.")

df_total = cargar_datos()
if df_total is not None:
    df_coordenadas = simular_coordenadas(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    # ... (tus filtros van aquí, sin cambios) ...
    st.sidebar.title("Filtros del Mapa")
    lista_categorias = ['Todas'] + sorted(list(df_total['Categoria'].unique()))
    cat_sel = st.sidebar.selectbox("Filtrar por Categoría:", options=lista_categorias, index=0, key='mapa_categoria')
    metrica_sel = st.sidebar.radio("Métrica a Visualizar:", options=['Total Ventas (€)', 'Total Unidades'], key='mapa_metrica')
    fecha_min = df_total['Fecha'].min()
    fecha_max = df_total['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Selecciona un rango de fechas:", value=[fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max, key='mapa_fechas')

    # --- Aplicar Filtros ---
    # ... (tu lógica de filtrado va aquí, sin cambios) ...
    if len(rango_fechas) == 2:
        df_filtrado = df_total[(df_total['Fecha'] >= rango_fechas[0]) & (df_total['Fecha'] <= rango_fechas[1])]
        if cat_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Categoria'] == cat_sel]
    else:
        df_filtrado = df_total.copy()

    # --- AGREGACIÓN DE DATOS PARA EL MAPA ---
    # ... (tu lógica de agregación y cálculo de df_mapa va aquí, sin cambios) ...
    df_agg_farmacia = df_filtrado.groupby('Farmacia_ID').agg(
        Total_Ventas_Euros=('Total_Venta_€', 'sum'),
        Total_Unidades=('Cantidad', 'sum')
    ).reset_index()
    df_mapa = df_coordenadas.merge(df_agg_farmacia, on='Farmacia_ID', how='left').fillna(0)
    df_mapa['Ventas_str'] = df_mapa['Total_Ventas_Euros'].apply(lambda x: f"{x:,.0f} €")
    df_mapa['Unidades_str'] = df_mapa['Total_Unidades'].apply(lambda x: f"{x:,.0f}")
    zona_color_map = {'Centro': [0, 191, 255, 180], 'Norte': [50, 205, 50, 180], 'Sur': [255, 69, 0, 180]}
    df_mapa['color'] = df_mapa['Zona_Farmacia'].apply(lambda z: zona_color_map.get(z, [128, 128, 128, 160]))
    if metrica_sel == 'Total Ventas (€)': columna_size = 'Total_Ventas_Euros'
    else: columna_size = 'Total_Unidades'
    max_val = df_mapa[columna_size].max()
    if max_val > 0: df_mapa['radius_meters'] = (df_mapa[columna_size] / max_val) * 500 + 100
    else: df_mapa['radius_meters'] = 100

    st.header(f"Mostrando: {metrica_sel} para '{cat_sel}'")

    if df_mapa.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        # --- INICIO DE LA DEPURACIÓN ---
        st.subheader("🕵️ Datos para el Mapa (Depuración)")
        st.markdown("Verifica que las columnas `lat`, `lon`, `radius_meters` sean numéricas y no tengan valores nulos (NaN).")
        st.write("Tipos de datos:")
        st.dataframe(df_mapa.dtypes.astype(str)) # Muestra los tipos de datos
        st.write("Valores Nulos:")
        st.dataframe(df_mapa.isna().sum()) # Muestra la cuenta de nulos por columna
        st.dataframe(df_mapa) # Muestra la tabla completa
        st.divider()
        # --- FIN DE LA DEPURACIÓN ---

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
            r = pdk.Deck(
                layers=[scatterplot_layer, text_layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/dark-v9',
                tooltip=tooltip
            )

            # Renderizar el mapa en Streamlit
            st.pydeck_chart(r, use_container_width=True) # <-- Corregido el aviso de obsolescencia

            st.subheader("Datos Detallados del Mapa")
            # Muestra la tabla original (sin las columnas de depuración si quieres)
            st.dataframe(df_mapa[['Farmacia_ID', 'Zona_Farmacia', 'Total_Ventas_Euros', 'Total_Unidades', 'lat', 'lon']], use_container_width=True)

        except Exception as e:
            st.error("💥 ¡Error al renderizar el mapa Pydeck!")
            st.error(f"Detalles: {e}")
            st.warning("Revisa los datos de depuración de arriba. ¿Hay algún NaN o tipo de dato incorrecto?")

else:
    st.error("Error al cargar los datos.")
