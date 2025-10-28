import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- MEJORA VISUAL: Configuración de Pestaña ---
st.set_page_config(page_title="Segmentación", page_icon="🧬", layout="wide")

# --- MEJORA FUNCIONAL: Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- FUNCIONES DE DATOS (CON CACHÉ) ---
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
def simular_y_calcular_rentabilidad(df_in):
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
    df_perfil = _df_rentabilidad.groupby('Farmacia_ID').agg(
        Total_Ventas_Euros=('Total_Venta_€', 'sum'),
        Total_Rentabilidad_Euros=('Margen_Neto_€', 'sum')
    ).reset_index()
    df_ventas_categoria = _df_rentabilidad.groupby(['Farmacia_ID', 'Categoria'])['Total_Venta_€'].sum().unstack(fill_value=0)
    df_ventas_categoria_pct = df_ventas_categoria.div(df_ventas_categoria.sum(axis=1), axis=0)
    df_ventas_categoria_pct = df_ventas_categoria_pct.rename(columns={
        c: f"Pct_Ventas_{c.replace(' ', '_')}" for c in df_ventas_categoria_pct.columns
    })
    df_perfil_final = df_perfil.merge(df_ventas_categoria_pct, on='Farmacia_ID', how='left')
    return df_perfil_final

@st.cache_resource
def entrenar_modelo_kmeans(df_perfil_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_perfil_scaled)
    return kmeans.labels_

# --- INTERFAZ DE STREAMLIT ---
st.title("🧬 Segmentación de Farmacias (Clustering)")
st.info("""
**💡 ¿Para qué sirve esto?**
Esta página agrupa automáticamente tus farmacias en **'segmentos'** (ej. "Alto Crecimiento", "Enfocadas en Invierno") basándose en su rendimiento y tipo de ventas. 
Te permite crear **estrategias de marketing y stock personalizadas** para cada grupo. Es una herramienta de **Machine Learning No Supervisado (KMeans)**.
""", icon="💡")
st.markdown("### ¿Podemos agrupar nuestras farmacias en 'segmentos' con comportamientos similares?")

df_total = cargar_datos()
if df_total is not None:
    df_rentabilidad = simular_y_calcular_rentabilidad(df_total)
    df_perfil = crear_perfil_farmacias(df_rentabilidad)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Parámetros del Modelo (KMeans)")
    n_clusters = st.sidebar.slider(
        "Número de Segmentos (Clusters):", 2, 5, 3,
        help="¿Cuántos grupos distintos de farmacias quieres que la IA intente encontrar?"
    )
    st.sidebar.divider()
    
    # --- PREPROCESAMIENTO Y MODELADO ---
    features = df_perfil.columns.drop('Farmacia_ID')
    X = df_perfil[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = entrenar_modelo_kmeans(X_scaled, n_clusters)
    df_perfil['Cluster'] = labels
    df_perfil['Cluster'] = df_perfil['Cluster'].astype(str)

    # --- VISUALIZACIÓN DE RESULTADOS ---
    st.header(f"Visualización de {n_clusters} Segmentos de Farmacias")
    chart = alt.Chart(df_perfil).mark_circle(size=100).encode(
        x=alt.X('Total_Ventas_Euros', title='Ventas Totales (€)', scale=alt.Scale(zero=False)),
        y=alt.Y('Total_Rentabilidad_Euros', title='Rentabilidad Total (€)', scale=alt.Scale(zero=False)),
        color=alt.Color('Cluster', title="Segmento"),
        tooltip=['Farmacia_ID', 'Cluster', 'Total_Ventas_Euros', 'Total_Rentabilidad_Euros']
    ).properties(title="Segmentación de Farmacias por Ventas y Rentabilidad").interactive()
    st.altair_chart(chart, use_container_width=True)
    st.divider()
    
    # --- INTERPRETACIÓN DE CLUSTERS ---
    st.header("¿Qué significa cada segmento?")
    st.markdown("Promedio de las métricas para cada segmento (normalizadas de 0 a 1 para comparar).")
    df_perfil_scaled_con_cluster = pd.DataFrame(X_scaled, columns=features)
    df_perfil_scaled_con_cluster['Cluster'] = labels
    df_interpretacion = df_perfil_scaled_con_cluster.groupby('Cluster').mean().reset_index()
    st.dataframe(df_interpretacion.set_index('Cluster'), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Datos Detallados por Farmacia")
    st.dataframe(df_perfil.sort_values(by="Cluster"), use_container_width=True)
    
    # --- MEJORA: BOTÓN DE DESCARGA ---
    csv_data = convert_df_to_csv(df_perfil)
    st.download_button(
        label=" Descargar Datos de Segmentación en CSV",
        data=csv_data,
        file_name=f"reporte_segmentacion_farmacias.csv",
        mime='text/csv',
        use_container_width=True
    )
else:
    st.error("Error al cargar los datos.")