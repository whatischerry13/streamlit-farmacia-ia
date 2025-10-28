import streamlit as st

# --- Configuración de Página ---
st.set_page_config(
    page_title="Portfolio IA Farmacia | [Alberto Tejero Castro]",
    page_icon="🧬",
    layout="wide"
)

# --- INICIO DE LA CORRECCIÓN ---

# 1. Logo en la Barra Lateral
# (Asegúrate de tener un logo.png en tu carpeta)
try:
    # Corregido: 'use_container_width' es el parámetro moderno
    st.sidebar.image("logo.png", use_container_width=True) 
except:
    st.sidebar.title("⚕️ Portfolio IA Farmacia") # Fallback si no hay logo

# --- FIN DE LA CORRECCIÓN ---



# --- Contenido de la Página ---

# Encabezado
st.title("Bienvenido al Dashboard de Inteligencia Artificial para Farmacias")
st.markdown("### Un proyecto de portfolio de [Alberto Tejero Castro]")
st.divider()

# Dividir la intro para poner un GIF
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Propósito del Proyecto")
    st.markdown("""
    Esta aplicación web interactiva es un **prototipo de nivel profesional** diseñado para demostrar las capacidades del análisis de datos y la inteligencia artificial en el sector farmacéutico. 

    El objetivo es transformar datos brutos de ventas (`CSV`) en una herramienta estratégica que permita a los equipos de **Ventas, Marketing y Operaciones** tomar decisiones informadas.
    """)
    
    st.subheader("🛠️ Stack Tecnológico")
    st.markdown("""
    - **🐍 Python:** `Pandas`, `Numpy`
    - **🤖 Machine Learning:** `XGBoost`, `Scikit-learn (KMeans)`, `mlxtend`, `statsmodels`
    - **📊 Visualización:** `Streamlit`, `Altair`, `Pydeck`
    - **⚙️ MLOps (Simulado):** `Joblib` para la separación de entrenamiento/inferencia.
    """)


# Sobre Mí
st.header("🧑‍💻 Sobre Mí")
st.markdown("""
¡Hola! Soy **[Alberto]**, un desarrollador y analista de datos apasionado por [ el Machine Learning, la visualización de datos y la creación de productos de software].

Este proyecto demuestra mi dominio en todo el ciclo de vida de un producto de datos:
""")
st.markdown("""
- **Análisis de Datos:** Limpieza, procesamiento (`Pandas`) y análisis estadístico (`Statsmodels`).
- **Machine Learning:**
    - **Supervisado:** Pronóstico de series temporales con `XGBoost`.
    - **No Supervisado:** Segmentación de clientes/farmacias con `KMeans`.
    - **Reglas de Asociación:** Análisis de cesta de compra con `mlxtend (Apriori)`.
- **Visualización de Datos:** Creación de dashboards interactivos con `Streamlit`, `Altair` y mapas 3D con `Pydeck`.
- **Ingeniería de ML (MLOps):** Optimización de rendimiento separando el entrenamiento (`train_models.py`) de la inferencia (`.joblib`), un pilar de la puesta en producción.
""")
st.markdown("[Mi Perfil de LinkedIn](https://www.linkedin.com/in/alberto-tejero-castro-a7847a294/)")


st.divider()

# Cómo Usar la App
st.header("🧭 Cómo Usar este Dashboard")
st.info("Usa el **menú de la izquierda** para navegar por las diferentes secciones de análisis:")

st.subheader("📈 Resumen General")
st.markdown("Visión general de KPIs, análisis de estacionalidad (Alergias vs. Antigripales) y un pronóstico de IA interactivo.")

st.subheader("🛒 Análisis de Cesta")
st.markdown("Descubre qué productos se compran juntos usando el algoritmo *Apriori*.")

st.subheader("💰 Rentabilidad")
st.markdown("Identifica qué productos y farmacias generan el mayor margen de beneficio, no solo las mayores ventas.")

st.subheader("🚨 Alerta de Stock (IA)")
st.markdown("El núcleo de la app: un sistema proactivo que usa la IA para predecir la demanda futura y compararla con el stock (simulado) para evitar roturas.")

st.subheader("⏰ Horas Pico")
st.markdown("Analiza los patrones de venta por hora para optimizar los turnos de personal.")

st.subheader("🗺️ Mapa de Ventas")
st.markdown("Visualización geoespacial 3D (usando `Pydeck`) del rendimiento de las farmacias por zona.")

st.subheader("🧬 Segmentación (IA)")
st.markdown("Agrupa farmacias en 'clusters' según su comportamiento de ventas (`KMeans`).")

st.subheader("🧪 Simulador Escenarios")
st.markdown("Herramienta de análisis prescriptivo para simular el impacto de campañas de marketing.")