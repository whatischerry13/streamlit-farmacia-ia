import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. Configuración de Página ---
st.set_page_config(page_title="Segmentación de Farmacias", layout="wide")

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
def simular_y_calcular_rentabilidad(df_in):
    """(Copiado de la pág. Rentabilidad)"""
    df = df_in.copy()
    productos = df[['Producto', 'Precio_Unitario_€']].drop_duplicates()
    np.random.seed(42)
    ratios_coste = np.random.uniform(0.4, 0.8, size=len(productos))
    productos['Ratio_Coste'] = ratios_coste
    productos['Coste_Unitario_€'] = productos['Precio_Unitario_€'] * productos['Ratio_Coste']
    df = df.merge(productos[['Producto', 'Coste_Unitario_€']], on='Producto', how='left')
    df['Margen_Unitario_€'] = df['Precio_Unitario_€'] - df['Coste_Unitario_€']
    df['Margen_Neto_€'] = df['Margen_Unitario_€'] * df['Cantidad']
    return df

@st.cache_data
def crear_perfil_farmacias(_df_rentabilidad):
    """
    Crea el DataFrame de "features" para el clustering.
    Un perfil por cada farmacia.
    """
    df_perfil = _df_rentabilidad.groupby('Farmacia_ID').agg(
        Ventas_Totales=('Total_Venta_€', 'sum'),
        Rentabilidad_Neta=('Margen_Neto_€', 'sum')
    ).reset_index()
    
    df_ventas_categoria = _df_rentabilidad.groupby(['Farmacia_ID', 'Categoria'])['Total_Venta_€'].sum().unstack(fill_value=0)
    df_ventas_categoria_pct = df_ventas_categoria.div(df_ventas_categoria.sum(axis=1), axis=0)
    
    # Renombrar columnas para que sean más legibles
    df_ventas_categoria_pct = df_ventas_categoria_pct.rename(columns={
        c: f"Pct. {c.replace('_', ' ')}" for c in df_ventas_categoria_pct.columns
    })
    
    df_perfil_final = df_perfil.merge(df_ventas_categoria_pct, on='Farmacia_ID', how='left')
    
    return df_perfil_final.rename(columns={
        'Ventas_Totales': 'Ventas Totales',
        'Rentabilidad_Neta': 'Rentabilidad Neta'
    })

@st.cache_resource
def entrenar_modelo_kmeans(df_perfil_scaled, n_clusters):
    """
    Entrena el modelo KMeans y devuelve las etiquetas.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_perfil_scaled)
    return kmeans.labels_

# --- INTERFAZ DE STREAMLIT ---
st.title("Segmentación de Farmacias (Clustering)")
st.info("""
**¿Para qué sirve esto?**
Esta página agrupa automáticamente tus farmacias en **'segmentos'** (ej. "Alto Crecimiento", "Enfocadas en Invierno") basándose en su rendimiento y tipo de ventas. 
Permite crear **estrategias de marketing y stock personalizadas** para cada grupo. Es una herramienta de **Machine Learning No Supervisado (KMeans)**.
""", icon="ℹ️")

df_total = cargar_datos()
if df_total is not None:
    df_rentabilidad = simular_y_calcular_rentabilidad(df_total)
    df_perfil = crear_perfil_farmacias(df_rentabilidad)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Parámetros del Modelo (KMeans)")
    
    n_clusters = st.sidebar.slider(
        "Número de Segmentos (Clusters):",
        min_value=2, max_value=4, value=3, # Limitamos a 4 para que quepa en las columnas
        help="¿Cuántos grupos distintos de farmacias quieres que la IA intente encontrar?"
    )
    
    st.sidebar.divider()
    
    # --- PREPROCESAMIENTO Y MODELADO ---
    features = df_perfil.columns.drop('Farmacia_ID')
    X = df_perfil[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = entrenar_modelo_kmeans(X_scaled, n_clusters)
    df_perfil['Segmento'] = labels.astype(str) # Renombrado a "Segmento"

    # --- VISUALIZACIÓN DE RESULTADOS ---
    
    st.header("Personalidad de cada Segmento")
    
    # 1. Preparar los datos para el heatmap
    df_perfil_scaled_con_cluster = pd.DataFrame(X_scaled, columns=features)
    df_perfil_scaled_con_cluster['Segmento'] = labels.astype(str)
    
    # Calcular la media de cada feature (escalada) por cluster
    df_interpretacion = df_perfil_scaled_con_cluster.groupby('Segmento').mean().reset_index()
    
    # "Fundir" (Melt) el dataframe para que sea compatible con Altair
    df_heatmap = df_interpretacion.melt(
        id_vars='Segmento', 
        var_name='Metrica', 
        value_name='Valor_Escalado'
    )
    
    # 2. Crear el Heatmap
    heatmap = alt.Chart(df_heatmap).mark_rect().encode(
        x=alt.X('Metrica:N', title='Métrica de Perfil', sort=None), # N para Nominal (categórico)
        y=alt.Y('Segmento:O', title='Segmento'), # O para Ordinal (categórico ordenado)
        
        # El color usará una escala "divergente" (azul-blanco-rojo)
        color=alt.Color('Valor_Escalado', 
                        title="Nivel vs. Media",
                        scale=alt.Scale(range='diverging', domainMid=0) # Centrado en 0
                       ),
        
        tooltip=[
            alt.Tooltip('Segmento', title='Segmento'),
            alt.Tooltip('Metrica', title='Métrica'),
            alt.Tooltip('Valor_Escalado', title='Valor (Desv. de la Media)', format='.2f')
        ]
    ).properties(
        title="Mapa de Calor de la Personalidad de cada Segmento"
    ).interactive()
    
    st.altair_chart(heatmap, use_container_width=True)
    
    st.info("""
    **Cómo leer este Mapa de Calor:**
    * **Rojo:** Este segmento está **muy por encima de la media** en esta métrica.
    * **Azul:** Este segmento está **muy por debajo de la media** en esta métrica.
    * **Blanco/Gris:** Este segmento está *en* la media.
    """)
    
    st.divider()

    # --- MEJORA: "Tarjetas de Personalidad" (Interpretación Automática) ---
    st.header("Interpretación de los Segmentos")
    st.markdown("Un resumen automático de la característica más destacada de cada segmento.")

    cols = st.columns(n_clusters) # Crea 2, 3 o 4 columnas
    
    # Ponemos el índice en 'Segmento' para buscar fácil
    df_interpretacion_idx = df_interpretacion.set_index('Segmento')
    
    for i in range(n_clusters):
        segmento_str = str(i)
        with cols[i]:
            st.subheader(f"Segmento {segmento_str}")
            
            # 1. Contar cuántas farmacias hay
            count = df_perfil[df_perfil['Segmento'] == segmento_str].shape[0]
            st.metric(label="Número de Farmacias", value=count)
            
            # 2. Obtener el perfil de este segmento
            profile = df_interpretacion_idx.loc[segmento_str]
            
            # 3. Encontrar la característica más alta y más baja
            # Usamos .drop() por si 'Ventas Totales' y 'Rentabilidad Neta' son siempre las más altas
            # y queremos ver la "personalidad" (los porcentajes)
            try:
                profile_pct_only = profile.drop(['Ventas Totales', 'Rentabilidad Neta'])
                highest_feature = profile_pct_only.idxmax()
                lowest_feature = profile_pct_only.idxmin()
            except: # Fallback si solo hay 2 features
                highest_feature = profile.idxmax()
                lowest_feature = profile.idxmin()

            # 4. Mostrar las tarjetas de perfil
            st.markdown(f"**Característica Principal:**")
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: #006400; color: white; margin-bottom: 10px;'>{highest_feature}</div>", unsafe_allow_html=True)

            st.markdown(f"**Punto Débil Principal:**")
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: #9B2B2B; color: white;'>{lowest_feature}</div>", unsafe_allow_html=True)
    # --- FIN DE LA MEJORA ---
    
    st.markdown("---")
    st.subheader("Asignación de Farmacias por Segmento")
    st.dataframe(df_perfil.sort_values(by="Segmento"), use_container_width=True)
    
    # --- 3. Botón de Descarga ---
    csv_data = convert_df_to_csv(df_perfil)
    st.download_button(
        label="Descargar Datos de Segmentación en CSV",
        data=csv_data,
        file_name=f"reporte_segmentacion_farmacias.csv",
        mime='text/csv',
        use_container_width=True
    )

else:
    st.error("Error al cargar los datos.")