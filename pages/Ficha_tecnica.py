import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Ficha Técnica", layout="wide")

# --- CARGA DE DATOS ---
@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        return None

# --- FUNCIONES DE NORMALIZACIÓN ---
def traducir_variables(df_importancia):
    traducciones = {
        'lag_1': 'Ventas (Ayer)', 
        'lag_2': 'Ventas (Anteayer)',
        'lag_7': 'Ventas (Hace 1 semana)',
        'lag_14': 'Ventas (Hace 2 semanas)',
        'roll_mean_7': 'Tendencia (7 días)',
        'roll_mean_28': 'Tendencia (28 días)',
        'roll_std_7': 'Volatilidad (7 días)',
        'tendencia_semanal': 'Inercia Semanal',
        'Temperatura_Media': 'Temperatura Media',
        'es_festivo': 'Es Festivo',
        'temp_gripe': 'Temporada Gripe',
        'temp_alergia': 'Temporada Alergia',
        'mes_sin': 'Ciclo Anual (Comp. A)',
        'mes_cos': 'Ciclo Anual (Comp. B)',
        'dia_semana_sin': 'Ciclo Semanal (Comp. A)',
        'dia_semana_cos': 'Ciclo Semanal (Comp. B)'
    }
    df = df_importancia.copy()
    df['Factor'] = df['Impulsor'].map(traducciones).fillna(df['Impulsor'])
    return df[['Factor', 'Importancia']]

def plot_radar(metrics):
    vals = [
        metrics.get('accuracy', 0),
        metrics.get('f1_score', 0),
        metrics.get('sensitivity', 0),
        metrics.get('precision', 0)
    ]
    labels = ['Accuracy', 'F1-Score', 'Sensibilidad', 'Precisión']
    
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill='toself',
        line_color='#2c3e50',
        fillcolor='rgba(44, 62, 80, 0.15)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='lightgrey')), 
        showlegend=False, 
        height=280, 
        margin=dict(t=20, b=20, l=40, r=40),
        font=dict(size=12, family="Arial")
    )
    return fig

# --- INICIO APLICACIÓN ---
st.title("Auditoría Técnica del Sistema")
st.markdown("Documentación viva: métricas de rendimiento, catálogo de algoritmos y arquitectura de datos.")

datos = cargar_modelos()

if not datos:
    st.error("No se encontraron modelos entrenados. Ejecute 'train_models.py'.")
    st.stop()

# --- NAVEGACIÓN ---
tab_global, tab_inspector, tab_catalogo, tab_arq = st.tabs([
    "Rendimiento Global", 
    "Inspector de Modelos", 
    "Catálogo de IA", 
    "Arquitectura"
])

# ==============================================================================
# TAB 1: RENDIMIENTO GLOBAL
# ==============================================================================
with tab_global:
    st.header("Estado del Ecosistema Predictivo")
    
    modelos = datos['modelos']
    n_modelos = len(modelos)
    df_metrics = pd.DataFrame([m['metrics'] for m in modelos.values()])
    
    # Explicación clara sobre "Qué modelos"
    st.info(f"""
    **Arquitectura de Micro-Modelos:** Este sistema no utiliza una única IA genérica. Se han entrenado **{n_modelos} modelos especialistas independientes**.
    
    Cada producto de cada farmacia tiene su propio modelo XGBoost entrenado exclusivamente con sus datos históricos. 
    Esto permite capturar la estacionalidad única de cada referencia (ej. un antigripal no se vende igual que una crema solar).
    """)
    
    st.divider()
    
    # KPIs Agregados
    st.subheader("Métricas Promedio del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precisión Global (Acc)", f"{df_metrics['accuracy'].mean():.1%}")
    c2.metric("Calidad Media (F1)", f"{df_metrics['f1_score'].mean():.2f}")
    c3.metric("Sensibilidad (Recall)", f"{df_metrics['sensitivity'].mean():.1%}")
    c4.metric("Error Medio (RMSE)", f"{df_metrics['rmse'].mean():.2f}")
    
    # Histograma
    st.divider()
    st.subheader("Distribución de Fiabilidad (Histograma F1-Score)")
    col_chart, col_txt = st.columns([2, 1])
    
    with col_chart:
        fig = px.histogram(df_metrics, x="f1_score", nbins=15, 
                          labels={'f1_score': 'Calidad del Modelo (F1 Score)'},
                          color_discrete_sequence=['#34495E'])
        fig.update_layout(yaxis_title="Cantidad de Modelos", bargap=0.1, height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_txt:
        st.markdown("**¿Qué significa este gráfico?**")
        st.markdown(f"""
        Visualiza la calidad de los {n_modelos} modelos simultáneamente.
        
        * **Barras a la derecha (0.8 - 1.0):** Indican productos donde la IA acierta casi siempre. Son predicciones muy fiables.
        * **Barras a la izquierda (< 0.6):** Indican productos "difíciles". Suelen ser artículos de venta muy esporádica donde es difícil encontrar un patrón.
        """)

# ==============================================================================
# TAB 2: INSPECTOR INDIVIDUAL
# ==============================================================================
with tab_inspector:
    st.header("Auditoría Individual XGBoost")
    st.markdown("Seleccione una combinación específica para inspeccionar su 'Caja Negra'.")
    
    claves = list(modelos.keys())
    farmacias = sorted(list(set([k.split('::')[0] for k in claves])))
    
    c_sel1, c_sel2 = st.columns(2)
    f_sel = c_sel1.selectbox("Farmacia:", farmacias)
    prods = sorted([k.split('::')[1] for k in claves if k.startswith(f_sel)])
    p_sel = c_sel2.selectbox("Producto:", prods)
    
    key_sel = f"{f_sel}::{p_sel}"
    modelo = modelos.get(key_sel)
    
    if modelo:
        st.markdown("---")
        col_izq, col_der = st.columns([1, 1.5])
        
        with col_izq:
            st.subheader("Perfil Métrico")
            st.plotly_chart(plot_radar(modelo['metrics']), use_container_width=True)
            
            st.markdown("**Resultados de Validación**")
            m = modelo['metrics']
            st.text(f"• RMSE (Error): {m['rmse']:.2f} uds.")
            st.text(f"• F1-Score:     {m['f1_score']:.2f}")
            st.text(f"• Accuracy:     {m['accuracy']:.1%}")

        with col_der:
            # Sección de Hiperparámetros (Top 4)
            st.subheader("Configuración del Algoritmo")
            params = modelo['model'].get_params()
            
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            c_p1.metric("Árboles (Estimators)", params.get('n_estimators'))
            c_p2.metric("Profundidad (Depth)", params.get('max_depth'))
            c_p3.metric("Tasa Aprendizaje", params.get('learning_rate'))
            c_p4.metric("Subsample", params.get('subsample'))
            
            st.markdown("---")
            
            # Sección de Importancia de Variables
            st.subheader("Factores de Decisión")
            st.caption("Variables que más influyen en la predicción de este producto específico.")
            
            if 'importance' in modelo:
                df_imp = traducir_variables(modelo['importance'].head(8))
                
                st.dataframe(
                    df_imp,
                    column_config={
                        "Factor": st.column_config.TextColumn("Variable", width="medium"),
                        "Importancia": st.column_config.ProgressColumn(
                            "Peso Relativo",
                            format="%.4f",
                            min_value=0,
                            max_value=df_imp['Importancia'].max(),
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

# ==============================================================================
# TAB 3: CATÁLOGO DE IA
# ==============================================================================
with tab_catalogo:
    st.header("Inventario de Algoritmos")
    st.markdown("Este proyecto despliega una arquitectura híbrida compuesta por **3 motores de inteligencia artificial** especializados:")
    
    st.divider()
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("1. Motor Predictivo (XGBoost)")
        st.info("""
        **Tipo:** Aprendizaje Supervisado (Regresión Avanzada)
        
        **Ubicación:** `Resumen General`, `Alerta de Stock`
        
        **Función:** Predecir la demanda futura diaria con alta precisión. Utiliza Gradient Boosting para aprender relaciones complejas no lineales entre el clima, el calendario y las ventas históricas. 
        Se entrena una instancia separada para cada producto.
        """)
        
        st.subheader("3. Motor de Asociación (Apriori)")
        st.warning("""
        **Tipo:** Minería de Datos (Reglas de Asociación)
        
        **Ubicación:** `Análisis de Cesta`
        
        **Función:** Detectar patrones de compra cruzada ocultos (Market Basket Analysis).
        Calcula la probabilidad condicional (Confianza) y la fuerza de la relación (Lift) entre productos para sugerir ventas cruzadas estratégicas.
        """)
        
    with c2:
        st.subheader("2. Motor de Segmentación (K-Means)")
        st.success("""
        **Tipo:** Aprendizaje No Supervisado (Clustering)
        
        **Ubicación:** `Segmentación`
        
        **Función:** Identificar arquetipos de farmacias automáticamente.
        El algoritmo analiza vectores multidimensionales (Ventas, Rentabilidad, Mix de Categorías) y agrupa las farmacias en clusters homogéneos sin necesidad de etiquetas previas, facilitando estrategias diferenciadas.
        """)

# ==============================================================================
# TAB 4: ARQUITECTURA DETALLADA
# ==============================================================================
with tab_arq:
    st.header("Pipeline de Datos (End-to-End)")
    st.markdown("Flujo técnico desde la ingesta de datos brutos hasta la inferencia en tiempo real.")
    
    # Diagrama Graphviz PROFESIONAL (Estilo Flat)
    st.graphviz_chart("""
    digraph {
        # Configuración Global
        rankdir=LR;
        bgcolor="transparent";
        splines=ortho;
        nodesep=0.8;
        ranksep=1.0;
        
        # Estilos de Nodos
        node [shape=box, style="filled,rounded", fontname="Sans-Serif", fontsize=10, penwidth=0, margin=0.2];
        edge [fontname="Sans-Serif", fontsize=9, color="#6c757d", penwidth=1.5];

        # Definición de Nodos con Colores Corporativos Suaves
        RAW [label="CSV Ventas\n+ Clima", fillcolor="#e9ecef", fontcolor="#495057"];
        
        subgraph cluster_preprocess {
            label = "1. Preprocesamiento (Feature Eng.)";
            fontname="Sans-Serif";
            fontsize=10;
            style=dashed;
            color="#adb5bd";
            
            FEAT [label="Transformación\nCíclica (Sin/Cos)", fillcolor="#d1e7dd", fontcolor="#0f5132"];
            LAGS [label="Lags & Rolling\nStats", fillcolor="#d1e7dd", fontcolor="#0f5132"];
        }

        subgraph cluster_training {
            label = "2. Entrenamiento (Offline)";
            fontname="Sans-Serif";
            fontsize=10;
            style=dashed;
            color="#adb5bd";
            
            SPLIT [label="TimeSeries\nSplit", fillcolor="#cfe2ff", fontcolor="#084298"];
            XGB [label="XGBoost\nRegressor", fillcolor="#cfe2ff", fontcolor="#084298"];
            OPT [label="Randomized\nSearch CV", fillcolor="#cfe2ff", fontcolor="#084298"];
        }

        STORE [label="Modelos.joblib\n(Serialización)", shape=folder, fillcolor="#fff3cd", fontcolor="#664d03"];
        
        subgraph cluster_app {
            label = "3. Inferencia (Online)";
            fontname="Sans-Serif";
            fontsize=10;
            style=dashed;
            color="#adb5bd";
            
            LOAD [label="Carga en Memoria\n(@st.cache)", fillcolor="#e2e3e5", fontcolor="#41464b"];
            PRED [label="Predicción\nRecursiva", fillcolor="#f8d7da", fontcolor="#842029"];
        }

        # Conexiones
        RAW -> FEAT;
        FEAT -> LAGS;
        LAGS -> SPLIT;
        SPLIT -> XGB;
        XGB -> OPT [dir=both, label="Optimización"];
        OPT -> STORE [label="Persistencia"];
        STORE -> LOAD;
        LOAD -> PRED;
    }
    """)
    
    st.divider()
    
    # Detalle Técnico Ampliado
    st.subheader("Profundización Técnica")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**1. Ingeniería de Variables (Feature Engineering)**")
        st.markdown("""
        Transformamos una serie temporal simple en un problema de aprendizaje supervisado rico en contexto:
        * **Codificación Temporal Cíclica:** El tiempo es circular, no lineal (después del mes 12 viene el 1). Utilizamos transformaciones trigonométricas (`sin(2π*m/12)`, `cos`) para preservar esta continuidad matemática y que la IA entienda la cercanía entre diciembre y enero.
        * **Memoria a Corto/Largo Plazo:** Generamos variables de retardo (`lag_1`, `lag_7`) para capturar la autocorrelación inmediata, y ventanas deslizantes (`rolling_mean_28`) para capturar la tendencia mensual subyacente.
        * **Contexto Externo:** Cruzamos los datos con calendarios de festivos nacionales y datos meteorológicos reales para enriquecer la capacidad predictiva.
        """)

    with c2:
        st.markdown("**2. Validación y Métrica Híbrida**")
        st.markdown("""
        El desafío técnico principal es alinear el objetivo matemático con el objetivo de negocio:
        * **Validación Robusta:** Utilizamos `TimeSeriesSplit` en lugar de `K-Fold` aleatorio. Esto es crítico para respetar la causalidad temporal y evitar el "data leakage" (predecir el pasado usando datos del futuro).
        * **Dualidad Regresión-Clasificación:** El modelo optimiza matemáticamente el `RMSE` (Error Cuadrático), pero para la toma de decisiones de stock, evaluamos su rendimiento mediante métricas de clasificación (`F1-Score`). Convertimos la predicción numérica en una señal binaria (Alta/Baja demanda) respecto a la media histórica para auditar si el modelo es útil detectando picos de venta.
        """)