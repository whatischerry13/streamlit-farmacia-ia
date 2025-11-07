import pandas as pd
import streamlit as st
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Configuración de Página ---
st.set_page_config(page_title="Análisis de Cesta", layout="wide")

# --- 2. Función de Descarga ---
@st.cache_data
def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# --- Funciones de Análisis ---
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
def simular_transacciones(df_in):
    """
    Simula IDs de transacción (Transaccion_ID) ya que no los tenemos.
    NOTA: Esta simulación introduce un sesgo significativo.
    """
    st.info("Simulando IDs de transacción... (Esto se ejecuta solo una vez)")
    
    PROMEDIO_ITEMS_POR_CESTA = 2.5
    df_con_id = []
    grupos = df_in.groupby(['Fecha', 'Farmacia_ID'])
    id_transaccion_global = 0
    
    for (fecha, farmacia), grupo_df in grupos:
        num_items_en_grupo = len(grupo_df)
        num_transacciones_estimadas = max(1, round(num_items_en_grupo / PROMEDIO_ITEMS_POR_CESTA))
        ids_transaccion_grupo = [f"T-{id_transaccion_global + i}" for i in range(num_transacciones_estimadas)]
        ids_asignados = np.random.choice(ids_transaccion_grupo, size=num_items_en_grupo, replace=True)
        grupo_df_copia = grupo_df.copy()
        grupo_df_copia['Transaccion_ID'] = ids_asignados
        df_con_id.append(grupo_df_copia)
        id_transaccion_global += num_transacciones_estimadas

    return pd.concat(df_con_id, ignore_index=True)

@st.cache_data
def preparar_datos_apriori(df, farmacia, categorias):
    """
    Prepara los datos para el algoritmo Apriori (formato one-hot).
    """
    if farmacia != 'Todas':
        df = df[df['Farmacia_ID'] == farmacia].copy()
    if categorias:
        df = df[df['Categoria'].isin(categorias)].copy()
    if df.empty:
        return None
    basket = df.groupby(['Transaccion_ID', 'Producto'])['Cantidad'].sum()
    basket_one_hot = basket.unstack(fill_value=0).applymap(lambda x: 1 if x > 0 else 0)
    return basket_one_hot

@st.cache_resource
def correr_analisis_apriori(basket_one_hot, min_support):
    """
    Ejecuta el algoritmo Apriori y las Reglas de Asociación.
    """
    frequent_itemsets = apriori(basket_one_hot, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return None
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    return rules

# --- INTERFAZ DE STREAMLIT ---
st.title("Análisis de Cesta de Compra")

st.info("""
**¿Para qué sirve esto?**
Esta sección utiliza un algoritmo de Machine Learning (Apriori) para analizar las transacciones (simuladas) y encontrar "reglas de asociación". 
Responde a la pregunta: *"El cliente que compró el Producto A, ¿qué más es probable que compre en la misma transacción?"*
""", icon="ℹ️")

# --- MEJORA PROFESIONAL: Advertencia sobre los datos ---
st.warning("""
**Nota sobre la Calidad de los Datos:**
Los resultados de esta simulación son limitados. El dataset original **no contiene IDs de transacción** (cestas de compra reales). 
Las "cestas" se han simulado agrupando ventas por día y farmacia, lo que introduce un sesgo que favorece reglas obvias (ej. productos más vendidos que aparecen juntos). 
En un proyecto real, la conexión al TPV (Terminal Punto de Venta) es esencial para obtener resultados precisos.
""")

df_total = cargar_datos()
if df_total is not None:
    df_transacciones = simular_transacciones(df_total)

    # --- Filtros ---
    st.sidebar.title("Filtros del Análisis")
    lista_farmacias = ['Todas'] + sorted(list(df_transacciones['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox("Filtrar por Farmacia:", options=lista_farmacias, key="cesta_farmacia")
    lista_categorias = sorted(list(df_transacciones['Categoria'].unique()))
    cat_sel = st.sidebar.multiselect("Filtrar por Categorías (Recomendado):", options=lista_categorias, default=['Alergia', 'Antigripal'], key="cesta_cat")
    
    st.sidebar.divider()
    st.sidebar.header("Parámetros del Modelo (IA)")
    
    # --- MEJORA DE UX: Arreglo del Slider ---
    # Cambiamos el valor por defecto 'value' de 0.01 a 0.003
    min_support = st.sidebar.slider(
        "Soporte Mínimo (Frecuencia):", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.003,  # <-- ¡VALOR POR DEFECTO CORREGIDO!
        step=0.001, 
        format="%.3f",
        help="Frecuencia mínima de un grupo de productos para ser analizado. Un valor más bajo encontrará más reglas."
    )

    # --- Resultados ---
    st.header(f"Resultados para: {farmacia_sel} (Categorías: {', '.join(cat_sel)})")
    basket_data = preparar_datos_apriori(df_transacciones, farmacia_sel, cat_sel)
    
    if basket_data is None or basket_data.empty:
        st.warning("No se encontraron transacciones con los filtros seleccionados.")
    else:
        with st.spinner(f"Analizando {len(basket_data)} cestas con soporte {min_support}..."):
            reglas = correr_analisis_apriori(basket_data, min_support)
        
        if reglas is None or reglas.empty:
            st.warning(f"No se encontraron reglas de asociación con un soporte de {min_support}. Prueba a bajar el 'Soporte Mínimo'.")
        else:
            st.success(f"¡Se encontraron {len(reglas)} reglas de asociación!")
            reglas_display = reglas.rename(columns={
                'antecedents': 'Si el cliente compra...',
                'consequents': '...también compra',
                'support': 'Soporte (Frec. Total)',
                'confidence': 'Confianza (Probabilidad)',
                'lift': 'Lift (Poder Predictivo)'
            })
            reglas_display['Si el cliente compra...'] = reglas_display['Si el cliente compra...'].apply(lambda x: ', '.join(list(x)))
            reglas_display['...también compra'] = reglas_display['...también compra'].apply(lambda x: ', '.join(list(x)))
            
            st.markdown("#### Reglas de Asociación Encontradas")
            st.info("""
            * **Confianza:** La probabilidad de que se compre "...también compra" *dado que* se compró "Si el cliente compra...". (Ej. 0.6 = 60% de las veces).
            * **Lift:** El poder predictivo de la regla. **Lift > 1** significa que es más probable que se compren juntos que por separado (esto es lo que buscamos).
            """)
            
            df_mostrar = reglas_display[['Si el cliente compra...', '...también compra', 'Confianza (Probabilidad)', 'Lift (Poder Predictivo)']]
            st.dataframe(df_mostrar, use_container_width=True)

            # --- Botón de Descarga ---
            csv_data = convert_df_to_csv(df_mostrar)
            st.download_button(
                label="Descargar Reglas de Asociación en CSV",
                data=csv_data,
                file_name=f"reglas_asociacion.csv",
                mime='text/csv',
                use_container_width=True
            )
else:
    st.error("Error al cargar los datos.")