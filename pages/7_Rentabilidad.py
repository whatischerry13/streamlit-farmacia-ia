import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

# --- 1. Configuración de Página (con layout="wide") ---
st.set_page_config(page_title="Rentabilidad", layout="wide")

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
    """
    Simula el 'Coste_Unitario_€' (basado en el precio) y calcula el margen.
    """
    df = df_in.copy()
    
    # Simulación de Costes (basada en el precio de venta para consistencia)
    productos = df[['Producto', 'Precio_Unitario_€']].drop_duplicates()
    np.random.seed(42) # Semilla para que los costes sean siempre los mismos
    ratios_coste = np.random.uniform(0.4, 0.8, size=len(productos))
    productos['Ratio_Coste'] = ratios_coste
    productos['Coste_Unitario_€'] = productos['Precio_Unitario_€'] * productos['Ratio_Coste']
    
    df = df.merge(productos[['Producto', 'Coste_Unitario_€']], on='Producto', how='left')
    
    # --- Cálculo de Rentabilidad ---
    df['Margen_Unitario_€'] = df['Precio_Unitario_€'] - df['Coste_Unitario_€']
    df['Margen_Neto_€'] = df['Margen_Unitario_€'] * df['Cantidad']
    
    return df

# --- INTERFAZ DE STREAMLIT ---
st.title("Análisis de Rentabilidad")

st.info("""
**¿Para qué sirve esto?**
Esta página va más allá de las ventas brutas. Analiza el **beneficio (margen)** de cada producto y farmacia.
Permite identificar qué productos son "vacas lecheras" (alto volumen, bajo margen) y cuáles son "joyas ocultas" (bajo volumen, alto margen).
""") # Usamos el 'st.info' neutro, sin icono

df_total = cargar_datos()
if df_total is not None:
    # --- MEJORA: Eliminamos el st.info molesto. La carga es silenciosa. ---
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

    # --- MEJORA: Organización por Pestañas ---
    tab1, tab2 = st.tabs(["Análisis de Producto", "Evolución Temporal de Rentabilidad"])
    
    if df_filtrado.empty:
        st.warning("No se encontraron datos con los filtros seleccionados.")
    else:
        with tab1:
            st.header("Análisis de Portfolio de Productos")
            
            # Agregamos los datos por producto
            df_productos_agg = df_filtrado.groupby(['Producto', 'Categoria']).agg(
                Ventas_Totales_Euros=('Total_Venta_€', 'sum'),
                Rentabilidad_Neta_Euros=('Margen_Neto_€', 'sum')
            ).reset_index()
            
            # Calcular Margen %
            df_productos_agg['Margen_Pct'] = (df_productos_agg['Rentabilidad_Neta_Euros'] / df_productos_agg['Ventas_Totales_Euros']) * 100
            df_productos_agg = df_productos_agg.fillna(0)

            # --- MEJORA: Gráfico de Matriz (Scatter Plot) ---
            st.subheader("Matriz de Posicionamiento (Ventas vs. Rentabilidad)")
            
            scatter_plot = alt.Chart(df_productos_agg).mark_circle(size=100).encode(
                x=alt.X('Ventas_Totales_Euros', title='Ventas Totales (€)'),
                y=alt.Y('Rentabilidad_Neta_Euros', title='Rentabilidad Neta (€)'),
                color=alt.Color('Categoria', title='Categoría'),
                size=alt.Size('Margen_Pct', title='Margen (%)', legend=alt.Legend(format='.0f')),
                tooltip=['Producto', 'Categoria', 'Ventas_Totales_Euros', 'Rentabilidad_Neta_Euros', alt.Tooltip('Margen_Pct', format='.1f')]
            ).properties(
                title="Posicionamiento de Productos"
            ).interactive()
            
            st.altair_chart(scatter_plot, use_container_width=True)
            st.info("""
            **Cómo leer este gráfico:**
            - **Arriba a la derecha (Estrellas):** Productos con altas ventas y alta rentabilidad. (Ej. Paracetamol)
            - **Abajo a la derecha (Vacas Lecheras):** Alto volumen de ventas, pero bajo margen de beneficio.
            - **Arriba a la izquierda (Joyas Ocultas):** Bajas ventas, pero cada venta es muy rentable.
            - **Abajo a la izquierda (Perros):** Bajas ventas y baja rentabilidad.
            """)
            
            st.divider()

            # --- Gráficos Top 10 (como antes) ---
            col_ventas, col_rentabilidad = st.columns(2)
            with col_ventas:
                st.subheader("Top 10 Productos por Ventas")
                chart_ventas = alt.Chart(df_productos_agg).mark_bar().encode(
                    x=alt.X('Ventas_Totales_Euros:Q', title="Ventas Totales (€)"),
                    y=alt.Y('Producto:N', sort='-x'),
                    color=alt.Color('Categoria', title='Categoría'),
                    tooltip=['Producto', 'Categoria', 'Ventas_Totales_Euros', 'Rentabilidad_Neta_Euros']
                ).transform_window(
                    rank='rank(Ventas_Totales_Euros)', sort=[alt.SortField('Ventas_Totales_Euros', order='descending')]
                ).transform_filter(alt.datum.rank <= 10).interactive()
                st.altair_chart(chart_ventas, use_container_width=True)

            with col_rentabilidad:
                st.subheader("Top 10 Productos por Rentabilidad")
                chart_rentabilidad = alt.Chart(df_productos_agg).mark_bar(color='#4682B4').encode( 
                    x=alt.X('Rentabilidad_Neta_Euros:Q', title="Rentabilidad Neta (€)"),
                    y=alt.Y('Producto:N', sort='-x'),
                    color=alt.Color('Categoria', title='Categoría'),
                    tooltip=['Producto', 'Categoria', 'Ventas_Totales_Euros', 'Rentabilidad_Neta_Euros']
                ).transform_window(
                    rank='rank(Rentabilidad_Neta_Euros)', sort=[alt.SortField('Rentabilidad_Neta_Euros', order='descending')]
                ).transform_filter(alt.datum.rank <= 10).interactive()
                st.altair_chart(chart_rentabilidad, use_container_width=True)
                
            st.markdown("---")
            st.subheader("Datos Detallados de Productos")
            st.dataframe(df_productos_agg.sort_values(by="Ventas_Totales_Euros", ascending=False), use_container_width=True)
            
            # --- Botón de Descarga ---
            csv_data = convert_df_to_csv(df_productos_agg)
            st.download_button(
                label="Descargar Datos de Rentabilidad de Productos en CSV",
                data=csv_data,
                file_name=f"reporte_rentabilidad_productos.csv",
                mime='text/csv',
                use_container_width=True
            )

        with tab2:
            st.header("Evolución Temporal de la Rentabilidad")
            
            # Preparamos los datos temporales
            df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Fecha'])
            df_rent_tiempo = df_filtrado.set_index('Fecha').groupby('Categoria').resample('M')['Margen_Neto_€'].sum().reset_index()
            df_rent_tiempo['Fecha'] = df_rent_tiempo['Fecha'] + pd.offsets.MonthEnd(0) # Asegurar fin de mes

            st.subheader("Rentabilidad Neta Mensual por Categoría")
            
            area_chart = alt.Chart(df_rent_tiempo).mark_area().encode(
                x=alt.X('Fecha:T', title='Mes'),
                y=alt.Y('Margen_Neto_€:Q', title='Rentabilidad Neta (€)', stack='zero'),
                color=alt.Color('Categoria', title='Categoría'),
                tooltip=['Fecha', 'Categoria', alt.Tooltip('Margen_Neto_€', title='Rentabilidad Neta', format=',.0f')]
            ).properties(
                title="Evolución de la Rentabilidad Mensual"
            ).interactive()
            
            st.altair_chart(area_chart, use_container_width=True)
            
            st.info("""
            Este gráfico muestra qué categorías generan más beneficio y en qué época del año. 
            (Ej. Se puede observar el pico de rentabilidad de 'Antigripal' en los meses de invierno).
            """)
            
            st.markdown("---")
            st.subheader("Datos Detallados de Rentabilidad Temporal")
            st.dataframe(df_rent_tiempo.sort_values(by="Fecha"), use_container_width=True)
            
            # --- Botón de Descarga ---
            csv_data_tiempo = convert_df_to_csv(df_rent_tiempo)
            st.download_button(
                label="Descargar Datos de Rentabilidad Temporal en CSV",
                data=csv_data_tiempo,
                file_name=f"reporte_rentabilidad_temporal.csv",
                mime='text/csv',
                use_container_width=True
            )
else:
    st.error("Error al cargar los datos.")