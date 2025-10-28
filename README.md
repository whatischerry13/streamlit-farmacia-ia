# 🧬 Dashboard de IA para Análisis de Ventas de Farmacia

Un dashboard de Business Intelligence y Machine Learning de nivel profesional construido con Streamlit y Python, diseñado como proyecto de portfolio.

**[➡️ Ver la App en Vivo](https://[TU ENLACE DE STREAMLIT AQUI].streamlit.app/)** _<-- ¡Reemplaza este enlace cuando lo despliegues!_

---

## 🎯 Propósito del Proyecto

Este proyecto demuestra la aplicación práctica del análisis de datos y la inteligencia artificial en el sector farmacéutico. Transforma datos de ventas (`CSV` simulado) en una herramienta interactiva para la toma de decisiones estratégicas en **Ventas, Marketing y Operaciones**.

---

## 🛠️ Stack Tecnológico Utilizado

* **Backend & Análisis:** Python, Pandas, Statsmodels
* **Machine Learning:** XGBoost (Forecasting), Scikit-learn (Clustering KMeans), mlxtend (Market Basket Apriori)
* **Frontend & Visualización:** Streamlit, Altair, Pydeck (Mapas 3D)
* **MLOps (Simulado):** Joblib (Separación de entrenamiento offline vs. inferencia online)

---

## ✨ Características Principales

Este dashboard multi-página incluye 9 módulos analíticos:

1.  **🏠 Bienvenida:** Portada del proyecto, descripción y guía de uso.
2.  **📈 Resumen General:** KPIs clave, análisis de estacionalidad (`statsmodels`), y pronóstico de demanda con `XGBoost`. Organizado con pestañas para una mejor UX.
3.  **🛒 Análisis de Cesta:** Descubre qué productos se compran juntos frecuentemente usando el algoritmo `Apriori`.
4.  **💰 Rentabilidad:** Identifica los productos y farmacias con mayor margen de beneficio (simulando costes).
5.  **🚨 Alerta de Stock (IA):** Sistema **reactivo** que compara la demanda predicha por IA con el stock (simulado) para evitar roturas inminentes.
6.  **⏰ Horas Pico:** Analiza los patrones de venta por hora del día (simulando timestamps).
7.  **🗺️ Mapa de Ventas:** Mapa 3D interactivo (`Pydeck`) que visualiza el rendimiento geográfico de las farmacias por zona y métrica.
8.  **🧬 Segmentación (IA):** Agrupa farmacias en "clusters" según su comportamiento de ventas usando `KMeans` y `StandardScaler`.
9.  **🧪 Simulador "What-If":** Herramienta de análisis **prescriptivo** para simular el impacto de escenarios de negocio (ej. campañas de marketing) sobre el inventario.

---

## 🚀 Cómo Ejecutar Localmente

1.  **Clonar el repositorio:**
    ```bash
    git clone [TU ENLACE DE GITHUB]
    cd [NOMBRE-DE-TU-CARPETA]
    ```
2.  **Crear un entorno virtual e instalar dependencias:**
    ```bash
    # Usando venv
    python -m venv .venv
    source .venv/bin/activate # o .\.venv\Scripts\activate en Windows
    pip install -r requirements.txt

    # O usando Conda
    conda create -n farmacia_env python=3.10
    conda activate farmacia_env
    pip install -r requirements.txt # O instalar manualmente con conda install
    ```
3.  **(Paso Crítico) Entrenar los modelos de IA:** Este script genera el archivo `modelos_farmacia.joblib`.
    ```bash
    python train_models.py
    ```
4.  **Ejecutar la aplicación Streamlit:**
    ```bash
    streamlit run 1_🏠_Bienvenida.py
    ```
    La aplicación se abrirá en tu navegador web.

---

## 🧑‍💻 Sobre Mí

¡Hola! Soy **[Alberto Tejero Castro]**, [un apasionado Analista de Datos con experiencia en Machine Learning...]. Me encanta transformar datos en soluciones prácticas y visualmente atractivas.

* **LinkedIn:** [https://www.linkedin.com/in/alberto-tejero-castro-a7847a294/]


_Este proyecto es una demostración de mis habilidades y está construido con datos simulados._"# streamlit-farmacia-ia" 
"# streamlit-farmacia-ia" 
