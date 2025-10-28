import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

# --- MEJORA VISUAL: Configuración de Pestaña ---
st.set_page_config(page_title="Rentabilidad", page_icon="💰", layout="wide")

# --- MEJORA FUNCIONAL: Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig') # UTF-8 con BOM para Excel

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

# --- INTERFAZ DE STREAMLIT ---
st.title("💰 Análisis de Rentabilidad")
st.info("💡 **¿Para qué sirve esto?** Esta página va más allá de las ventas. Identifica qué productos y farmacias generan el **mayor beneficio (margen)**, permitiéndote enfocar tus esfuerzos en lo que realmente da dinero.", icon="💡")

df_total = cargar_datos()
if df_total is not None:
    df_rentabilidad = simular_y_calcular_rentabilidad(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Filtros de Análisis")
    lista_farmacias = ['Todas'] + sorted(list(df_rentabilidad['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Filtrar por Farmacia:", options=lista_farmacias, key='rentabilidad_farmacia')
    lista_categorias = ['Todas'] + sorted(list(df_rentabilidad['Categoria'].unique()))
    cat_sel = st.sidebar.selectbox("Filtrar por Categoría:", options=lista_categorias, key='rentabilidad_categoria')
    fecha_min = df_rentabilidad['Fecha'].min()
    fecha_max = df_rentabilidad['Fecha'].max()
    rango_fechas = st.sidebar.date_input("Selecciona un rango de fechas:", value=[fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max, key='rentabilidad_fechas')
    
    # --- Aplicar Filtros ---
    if len(rango_fechas) == 2:
        df_filtrado = df_rentabilidad[
            (df_rentabilidad['Fecha'] >= rango_fechas[0]) &
            (df_rentabilidad['Fecha'] <= rango_fechas[1])
        ]
        if farmacia_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Farmacia_ID'] == farmacia_sel]
        if cat_sel != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Categoria'] == cat_sel]
    else:
        df_filtrado = df_rentabilidad.copy()

    # --- KPIs DE RENTABILIDAD ---
    st.header(f"Resultados para: {farmacia_sel} | {cat_sel}")
    total_ventas = df_filtrado['Total_Venta_€'].sum()
    total_rentabilidad = df_filtrado['Margen_Neto_€'].sum()
    margen_medio_pct = (total_rentabilidad / total_ventas) * 100 if total_ventas > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas Totales (€)", f"{total_ventas:,.2f} €")
    col2.metric("Rentabilidad Neta (€)", f"{total_rentabilidad:,.2f} €")
    col3.metric("Margen Medio (%)", f"{margen_medio_pct:,.1f} %")
    st.divider()

    # --- GRÁFICOS COMPARATIVOS ---
    st.header("Análisis de Productos")
    
    if df_filtrado.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        df_productos_agg = df_filtrado.groupby(['Producto', 'Categoria']).agg(
            Ventas_Totales=('Total_Venta_€', 'sum'),
            Rentabilidad_Neta=('Margen_Neto_€', 'sum')
        ).reset_index()

        col_ventas, col_rentabilidad = st.columns(2)
        with col_ventas:
            st.subheader("Top 10 Productos por Ventas")
            chart_ventas = alt.Chart(df_productos_agg).mark_bar().encode(
                x=alt.X('Ventas_Totales:Q', title="Ventas Totales (€)"),
                y=alt.Y('Producto:N', sort='-x'),
                tooltip=['Producto', 'Categoria', 'Ventas_Totales', 'Rentabilidad_Neta']
            ).transform_window(
                rank='rank(Ventas_Totales)', sort=[alt.SortField('Ventas_Totales', order='descending')]
            ).transform_filter(alt.datum.rank <= 10).interactive()
            st.altair_chart(chart_ventas, use_container_width=True)

        with col_rentabilidad:
            st.subheader("Top 10 Productos por Rentabilidad")
            chart_rentabilidad = alt.Chart(df_productos_agg).mark_bar(color='#00BFFF').encode( # Color primario del tema
                x=alt.X('Rentabilidad_Neta:Q', title="Rentabilidad Neta (€)"),
                y=alt.Y('Producto:N', sort='-x'),
                tooltip=['Producto', 'Categoria', 'Ventas_Totales', 'Rentabilidad_Neta']
            ).transform_window(
                rank='rank(Rentabilidad_Neta)', sort=[alt.SortField('Rentabilidad_Neta', order='descending')]
            ).transform_filter(alt.datum.rank <= 10).interactive()
            st.altair_chart(chart_rentabilidad, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Datos Detallados de Productos")
        st.dataframe(df_productos_agg.sort_values(by="Ventas_Totales", ascending=False), use_container_width=True)
        
        # --- MEJORA: BOTÓN DE DESCARGA ---
        csv_data = convert_df_to_csv(df_productos_agg)
        st.download_button(
            label="Descargar Datos de Productos en CSV",
            data=csv_data,
            file_name=f"reporte_rentabilidad_productos.csv",
            mime='text/csv',
            use_container_width=True
        )
else:
    st.error("Error al cargar los datos.")