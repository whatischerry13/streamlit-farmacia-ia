import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import warnings
import joblib
import holidays
import altair as alt
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Configuración ---
st.set_page_config(page_title="Simulador de Escenarios", layout="wide")

# --- 2. Funciones de Carga ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

@st.cache_data
def cargar_datos(file_name='ventas_farmacia_fake.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        # Renombrar columna de precio para evitar problemas con el símbolo € en iterros
        if 'Precio_Unitario_€' in df.columns:
            df = df.rename(columns={'Precio_Unitario_€': 'Precio'})
        return df
    except: return None

@st.cache_data
def cargar_clima(file_name='clima_madrid.csv'):
    try:
        df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
        return df
    except: return None

@st.cache_resource
def cargar_modelos(file_name='modelos_farmacia.joblib'):
    try: return joblib.load(file_name)
    except: return None

@st.cache_data
def simular_stock_actual(_df_total):
    st.info("Simulando inventario de stock actual... (se ejecuta una vez)")
    df_grp = _df_total.groupby(['Farmacia_ID', 'Producto'])['Cantidad'].mean().reset_index()
    np.random.seed(123)
    df_grp['Stock_Actual'] = (df_grp['Cantidad'] * np.random.randint(3, 20, size=len(df_grp))).astype(int)
    return df_grp[['Farmacia_ID', 'Producto', 'Stock_Actual']]

# --- LÓGICA DE PREDICCIÓN PREMIUM ---
def crear_features_un_paso(historia_y, fecha_obj, df_clima):
    row = pd.DataFrame({'ds': [pd.to_datetime(fecha_obj)]})
    mes = fecha_obj.month
    dia_sem = fecha_obj.weekday()
    
    row['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    row['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    row['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    row['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    
    es_holidays = holidays.Spain(years=[fecha_obj.year])
    row['es_festivo'] = int(fecha_obj in es_holidays)
    row['temp_gripe'] = int(mes in [10, 11, 12, 1, 2])
    row['temp_alergia'] = int(mes in [3, 4, 5, 6])
    
    t_media = 15.0
    if df_clima is not None:
        match = df_clima[df_clima['Fecha'] == fecha_obj]
        if not match.empty: t_media = match.iloc[0]['Temperatura_Media']
    row['Temperatura_Media'] = t_media
    
    vals = historia_y
    row['lag_1'] = vals[-1] if len(vals)>=1 else 0
    row['lag_2'] = vals[-2] if len(vals)>=2 else 0
    row['lag_7'] = vals[-7] if len(vals)>=7 else 0
    row['lag_14'] = vals[-14] if len(vals)>=14 else 0
    
    row['roll_mean_7'] = pd.Series(vals).rolling(7).mean().iloc[-1] if len(vals)>=7 else 0
    row['roll_mean_28'] = pd.Series(vals).rolling(28).mean().iloc[-1] if len(vals)>=28 else 0
    row['roll_std_7'] = pd.Series(vals).rolling(7).std().iloc[-1] if len(vals)>=7 else 0
    
    rm7 = row['roll_mean_7'].values[0]
    rm7_prev = pd.Series(vals).rolling(7).mean().iloc[-8] if len(vals)>=8 else rm7
    row['tendencia_semanal'] = row['roll_mean_7'] - rm7_prev
    
    cols = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
            'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal']
    return row[cols]

def predecir_recursivo(model, df_hist_prod, df_clima, dias_futuros, fecha_max):
    historia = list(df_hist_prod.sort_values('Fecha')['Cantidad'].values)
    predicciones = []
    for i in range(dias_futuros):
        fecha_futura = fecha_max + timedelta(days=i+1)
        X_test = crear_features_un_paso(historia, fecha_futura, df_clima)
        y_pred = max(0, model.predict(X_test)[0])
        predicciones.append(y_pred)
        historia.append(y_pred)
    return np.array(predicciones)

# --- APP ---
st.title("Simulador de Escenarios 'What-If'")
st.info("Análisis Prescriptivo: Simula el impacto de cambios en la demanda sobre el stock futuro.", icon="ℹ️")

df_total = cargar_datos(); df_clima = cargar_clima(); datos_modelos = cargar_modelos()

if df_total is not None and datos_modelos is not None:
    df_stock = simular_stock_actual(df_total)
    
    st.sidebar.header("Configuración")
    adj_gripe = st.sidebar.slider("Impacto Antigripal (%)", -50, 100, 0)
    adj_alergia = st.sidebar.slider("Impacto Alergia (%)", -50, 100, 0)
    
    st.sidebar.divider()
    farm_sel = st.sidebar.selectbox("Farmacia:", ['Todas'] + sorted(list(df_total['Farmacia_ID'].unique())))
    dias_sim = st.sidebar.slider("Días a Simular:", 7, 60, 14)
    
    if st.button("Ejecutar Simulación", type="primary", use_container_width=True):
        df_work = df_stock.copy()
        if farm_sel != 'Todas': df_work = df_work[df_work['Farmacia_ID'] == farm_sel]
        
        prods = df_total[['Producto', 'Categoria', 'Precio']].drop_duplicates(subset='Producto')
        df_work = df_work.merge(prods, on='Producto')
        df_work = df_work[df_work['Categoria'].isin(['Alergia', 'Antigripal'])]
        
        if df_work.empty:
            st.warning("Sin datos para simular.")
        else:
            resultados_tabla = []
            graficos_data = []
            
            fecha_max = df_total['Fecha'].max()
            modelos = datos_modelos['modelos']
            progreso = st.progress(0, text="Simulando...")
            
            # Usamos iterrows para evitar problemas de nombres de columna con símbolos
            for idx, row in df_work.iterrows():
                progreso.progress((idx+1)/len(df_work))
                
                farm_id = row['Farmacia_ID']
                prod = row['Producto']
                cat = row['Categoria']
                stock_ini = row['Stock_Actual']
                precio = row['Precio']
                
                key = f"{farm_id}::{prod}"
                m_info = modelos.get(key)
                if not m_info: continue
                
                df_h = df_total[(df_total['Farmacia_ID']==farm_id) & (df_total['Producto']==prod)]
                preds = predecir_recursivo(m_info['model'], df_h, df_clima, dias_sim, fecha_max)
                
                factor = 1 + (adj_gripe/100 if cat == 'Antigripal' else adj_alergia/100)
                preds_sim = preds * factor
                
                stock = stock_ini
                dia_rotura = "OK"
                stock_evolucion = []
                rotura_num = 999
                
                for d, venta in enumerate(preds_sim):
                    stock -= venta
                    stock_evolucion.append(stock)
                    if stock <= 0 and dia_rotura == "OK":
                        dia_rotura = f"{d+1} días"
                        rotura_num = d + 1
                
                prod_id = f"{prod} ({farm_id})"
                
                if dia_rotura != "OK":
                    # Guardamos gráfico
                    df_g = pd.DataFrame({
                        'Dia': range(1, dias_sim+1), 
                        'Stock': stock_evolucion, 
                        'ID': prod_id
                    })
                    graficos_data.append(df_g)
                    
                    # --- CREACIÓN SEGURA DEL DICCIONARIO ---
                    # Creamos el item paso a paso para evitar errores de sintaxis
                    item = {}
                    item["Farmacia"] = farm_id
                    item["Producto"] = prod
                    item["Día Rotura"] = dia_rotura
                    item["Stock Inicial"] = stock_ini
                    
                    valor_perdido = abs(stock) * precio
                    item["Ventas Perdidas (€)"] = f"{valor_perdido:.2f} €"
                    
                    item["ID"] = prod_id
                    item["Dias_Num"] = rotura_num
                    
                    resultados_tabla.append(item)
            
            progreso.empty()
            
            # MOSTRAR RESULTADOS
            fecha_inicio = (fecha_max + timedelta(days=1)).strftime('%d-%b-%Y')
            fecha_fin = (fecha_max + timedelta(days=dias_sim)).strftime('%d-%b-%Y')
            st.subheader(f"Horizonte: {fecha_inicio} al {fecha_fin}")
            
            col1, col2 = st.columns(2)
            col1.metric("Productos en Riesgo", len(resultados_tabla))
            
            if not resultados_tabla:
                st.success("✅ El stock soporta el escenario simulado.")
            else:
                df_res = pd.DataFrame(resultados_tabla)
                df_res = df_res.sort_values('Dias_Num')
                
                # Gráfico
                top5_ids = df_res.head(5)['ID'].tolist()
                
                if graficos_data:
                    df_plot = pd.concat(graficos_data)
                    df_plot = df_plot[df_plot['ID'].isin(top5_ids)]
                    
                    st.subheader("Evolución de Stock (Top 5 Riesgos)")
                    c = alt.Chart(df_plot).mark_line().encode(
                        x=alt.X('Dia', title='Día Simulación'), 
                        y=alt.Y('Stock', title='Unidades Restantes'),
                        color='ID', 
                        tooltip=['ID', 'Dia', 'Stock']
                    ).interactive()
                    linea = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='red').encode(y='y')
                    st.altair_chart(c + linea, use_container_width=True)
                
                # Tabla
                st.subheader("Detalle de Roturas")
                def estilo_rojo(s):
                    return ['background-color: #8B0000; color: white'] * len(s)
                
                cols_view = ["Farmacia", "Producto", "Día Rotura", "Stock Inicial", "Ventas Perdidas (€)"]
                st.dataframe(df_res[cols_view].style.apply(estilo_rojo, axis=1), use_container_width=True)
                st.download_button("Descargar CSV", convert_df_to_csv(df_res), "simulacion.csv", "text/csv")
else:
    st.error("Error de carga.")