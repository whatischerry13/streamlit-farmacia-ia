import pandas as pd
import streamlit as st
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Análisis de Cesta", page_icon="🛒", layout="wide")

# --- FUNCIONES DE ANÁLISIS (CON CACHÉ) ---

@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    """Carga los datos base desde el CSV."""
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        df['Fecha'] = df['Fecha'].dt.date
        return df
    except FileNotFoundError:
        st.error(f"¡ERROR! No se encontró el archivo '{file_name}'. Asegúrate de que esté en la carpeta principal.")
        return None

@st.cache_data
def simular_transacciones(df_in):
    """
    ¡Simulación! Crea IDs de transacción (Transaccion_ID) ya que no los tenemos.
    Agrupa productos por Farmacia y Día, y luego los asigna aleatoriamente
    a un número estimado de "cestas".
    """
    st.info("Simulando IDs de transacción... (Esto se ejecuta solo una vez)", icon="⏳")
    
    # Estimamos 2.5 productos por cesta de media
    PROMEDIO_ITEMS_POR_CESTA = 2.5
    
    df_con_id = []
    
    # Agrupamos por día y farmacia
    grupos = df_in.groupby(['Fecha', 'Farmacia_ID'])
    
    id_transaccion_global = 0
    
    for (fecha, farmacia), grupo_df in grupos:
        
        num_items_en_grupo = len(grupo_df)
        
        # Estimamos el número de transacciones para este grupo
        num_transacciones_estimadas = max(1, round(num_items_en_grupo / PROMEDIO_ITEMS_POR_CESTA))
        
        # Creamos los IDs de transacción (ej. 'T-100', 'T-101', ...)
        ids_transaccion_grupo = [f"T-{id_transaccion_global + i}" for i in range(num_transacciones_estimadas)]
        
        # Asignamos aleatoriamente cada item (fila) a uno de los IDs de transacción
        ids_asignados = np.random.choice(ids_transaccion_grupo, size=num_items_en_grupo, replace=True)
        
        grupo_df_copia = grupo_df.copy()
        grupo_df_copia['Transaccion_ID'] = ids_asignados
        
        df_con_id.append(grupo_df_copia)
        
        id_transaccion_global += num_transacciones_estimadas

    return pd.concat(df_con_id, ignore_index=True)


@st.cache_data
def preparar_datos_apriori(df, farmacia, categorias):
    """
    Prepara los datos para el algoritmo Apriori.
    Necesita un formato "one-hot": una fila por transacción, una columna por producto.
    """
    
    # Filtrar por farmacia
    if farmacia != 'Todas':
        df = df[df['Farmacia_ID'] == farmacia].copy()
    
    # Filtrar por categorías seleccionadas
    if categorias: # Si la lista no está vacía
        df = df[df['Categoria'].isin(categorias)].copy()

    if df.empty:
        return None

    # 1. Agrupar por cesta y producto (contamos cuántos de c/u hay en la cesta)
    basket = df.groupby(['Transaccion_ID', 'Producto'])['Cantidad'].sum()
    
    # 2. Convertir a "one-hot": nos importa si lo compró (1) o no (0)
    basket_one_hot = basket.unstack(fill_value=0).applymap(lambda x: 1 if x > 0 else 0)
    
    return basket_one_hot

@st.cache_resource # Usamos resource para el resultado del algo
def correr_analisis_apriori(basket_one_hot, min_support):
    """
    Ejecuta el algoritmo Apriori y las Reglas de Asociación.
    """
    # 1. Encontrar "itemsets" frecuentes (grupos de productos que superan el soporte)
    frequent_itemsets = apriori(basket_one_hot, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return None
    
    # 2. Generar las reglas (ej. {A} -> {B})
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Ordenar por "lift" (lo bien que A predice B) y "confidence" (la prob. de que B ocurra dado A)
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    return rules

# --- INTERFAZ DE STREAMLIT ---

st.title("🛒 Análisis de Cesta de Compra (Market Basket)")
st.markdown("""
Esta sección analiza qué productos se compran *juntos* en la misma transacción.
Nos ayuda a responder: "El cliente que compró el Producto A, ¿qué más es probable que compre?"
""")

# Cargar datos base
df_total = cargar_datos()

if df_total is not None:
    
    # 1. Simular Transacciones (usa el caché)
    df_transacciones = simular_transacciones(df_total)

    # --- FILTROS EN LA BARRA LATERAL ---
    st.sidebar.title("Filtros del Análisis")
    
    # Filtro de Farmacia
    lista_farmacias = ['Todas'] + sorted(list(df_transacciones['Farmacia_ID'].unique()))
    farmacia_sel = st.sidebar.selectbox(
        "Filtrar por Farmacia:",
        options=lista_farmacias
    )
    
    # Filtro de Categoría
    lista_categorias = sorted(list(df_transacciones['Categoria'].unique()))
    cat_sel = st.sidebar.multiselect(
        "Filtrar por Categorías (Recomendado):",
        options=lista_categorias,
        default=['Alergia', 'Antigripal'] # Por defecto filtramos las más interesantes
    )
    
    st.sidebar.divider()
    
    # --- PARÁMETROS DEL ALGORITMO ---
    st.sidebar.header("Parámetros del Modelo (IA)")
    
    # El "Soporte Mínimo" es el parámetro más importante.
    # Es la frecuencia mínima que debe tener un grupo de productos para ser considerado.
    # Valores bajos = más reglas, pero menos relevantes.
    # Valores altos = menos reglas, pero más fuertes.
    min_support = st.sidebar.slider(
        "Soporte Mínimo (Frecuencia):",
        min_value=0.001, # 0.1%
        max_value=0.1,   # 10%
        value=0.01,      # 1% por defecto
        step=0.001,
        format="%.3f",
        help="Define la frecuencia mínima de un grupo de productos para ser analizado. Un valor más bajo encontrará más reglas (pero menos significativas)."
    )

    # --- EJECUCIÓN Y RESULTADOS ---
    
    st.header(f"Resultados para: {farmacia_sel} (Categorías: {', '.join(cat_sel)})")
    
    # 1. Preparar datos para Apriori
    basket_data = preparar_datos_apriori(df_transacciones, farmacia_sel, cat_sel)
    
    if basket_data is None or basket_data.empty:
        st.warning("No se encontraron transacciones con los filtros seleccionados.")
    else:
        # 2. Ejecutar el análisis
        with st.spinner(f"Analizando {len(basket_data)} cestas con soporte {min_support}..."):
            reglas = correr_analisis_apriori(basket_data, min_support)
        
        if reglas is None or reglas.empty:
            st.warning(f"No se encontraron reglas de asociación con un soporte de {min_support}. Intenta bajar el 'Soporte Mínimo' en la barra lateral.")
        else:
            st.success(f"¡Se encontraron {len(reglas)} reglas de asociación!")
            
            # 3. Mostrar los resultados
            
            # Renombramos columnas para que sean claras para Ventas/Marketing
            reglas_display = reglas.rename(columns={
                'antecedents': 'Si el cliente compra...',
                'consequents': '...también compra',
                'support': 'Soporte (Frec. Total)',
                'confidence': 'Confianza (Probabilidad)',
                'lift': 'Lift (Poder Predictivo)'
            })
            
            # Limpiamos los "frozenset" para que se vea más limpio
            reglas_display['Si el cliente compra...'] = reglas_display['Si el cliente compra...'].apply(lambda x: ', '.join(list(x)))
            reglas_display['...también compra'] = reglas_display['...también compra'].apply(lambda x: ', '.join(list(x)))
            
            st.markdown("#### Reglas de Asociación Encontradas")
            st.info("""
            * **Confianza:** La probabilidad de que se compre "Y" si se compró "X". (Ej. 0.6 = 60% de las veces).
            * **Lift:** El poder predictivo de la regla. **Lift > 1** significa que es más probable que se compren juntos que por separado (¡esto es lo que buscamos!).
            """)
            
            # Mostramos la tabla con las columnas más importantes
            st.dataframe(reglas_display[[
                'Si el cliente compra...', 
                '...también compra', 
                'Confianza (Probabilidad)', 
                'Lift (Poder Predictivo)'
            ]], use_container_width=True)