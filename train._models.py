import pandas as pd
import numpy as np
import xgboost as xgb
import joblib # <-- Nueva importación
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constantes ---
ARCHIVO_DATOS = 'ventas_farmacia_fake.csv'
ARCHIVO_MODELOS_SALIDA = 'modelos_farmacia.joblib'
CATEGORIAS_A_ENTRENAR = ['Alergia', 'Antigripal'] # Solo entrenamos las importantes

# --- Funciones (copiadas de la app) ---

def cargar_datos(file_name):
    """Carga y prepara los datos iniciales desde el CSV."""
    print(f"Cargando datos de '{file_name}'...")
    df = pd.read_csv(
        file_name,
        delimiter=';',
        decimal=',',
        parse_dates=['Fecha']
    )
    df['Fecha'] = df['Fecha'].dt.date # Nos aseguramos que es solo fecha
    return df

def crear_caracteristicas_temporales(df_in):
    """Crea características de tiempo para el modelo de ML."""
    df_out = df_in.copy()
    df_out['ds'] = pd.to_datetime(df_out['ds']) # Aseguramos que sea datetime
    df_out['mes'] = df_out['ds'].dt.month
    df_out['dia_del_ano'] = df_out['ds'].dt.dayofyear
    df_out['semana_del_ano'] = df_out['ds'].dt.isocalendar().week.astype(int)
    df_out['dia_de_la_semana'] = df_out['ds'].dt.dayofweek
    df_out['ano'] = df_out['ds'].dt.year
    df_out['trimestre'] = df_out['ds'].dt.quarter
    return df_out

def entrenar_modelo_para_serie(df_serie):
    """
    Entrena un único modelo XGBoost para una serie de tiempo (un producto/farmacia).
    """
    df_diario = df_serie.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    
    # Si no hay datos suficientes, no se puede entrenar
    if df_diario.empty or len(df_diario) < 100: 
        return None 

    df_preparado = crear_caracteristicas_temporales(df_diario)
    
    features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
    target = 'y'
    X = df_preparado[features]
    y = df_preparado[target]

    # Usamos los parámetros optimizados
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=50, random_state=42
    )

    eval_size = int(len(X) * 0.2)
    if eval_size < 50: 
        model.fit(X, y, verbose=False)
    else:
        X_train, X_eval = X.iloc[:-eval_size], X.iloc[-eval_size:]
        y_train, y_eval = y.iloc[:-eval_size], y.iloc[-eval_size:]
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    
    return model

# --- BLOQUE PRINCIPAL DE ENTRENAMIENTO ---

def main():
    print("--- INICIANDO ENTRENAMIENTO OFFLINE DE MODELOS DE IA ---")
    start_time = datetime.now()
    
    df_total = cargar_datos(ARCHIVO_DATOS)
    
    # 1. Obtener todas las combinaciones únicas de Farmacia y Producto
    df_total_filtrado = df_total[df_total['Categoria'].isin(CATEGORIAS_A_ENTRENAR)]
    combinaciones = df_total_filtrado[['Farmacia_ID', 'Producto']].drop_duplicates()
    
    print(f"Se entrenarán modelos para {len(combinaciones)} combinaciones (Producto/Farmacia)...")
    
    modelos_entrenados = {}
    modelos_fallidos = 0
    
    # 2. Iterar por cada combinación y entrenar un modelo
    for idx, row in combinaciones.iterrows():
        farmacia_id = row['Farmacia_ID']
        producto = row['Producto']
        
        # Creamos una clave única para el diccionario
        clave_modelo = f"{farmacia_id}::{producto}" 
        
        print(f"  Entrenando: {clave_modelo} ({idx+1}/{len(combinaciones)})")
        
        # Filtramos los datos solo para esta serie
        df_serie = df_total_filtrado[
            (df_total_filtrado['Farmacia_ID'] == farmacia_id) &
            (df_total_filtrado['Producto'] == producto)
        ]
        
        modelo = entrenar_modelo_para_serie(df_serie)
        
        if modelo:
            modelos_entrenados[clave_modelo] = modelo
        else:
            print(f"    -> Fallido (datos insuficientes)")
            modelos_fallidos += 1
            
    # 3. Guardar el diccionario de modelos en un solo archivo
    print("\nEntrenamiento completado.")
    print(f"  Modelos exitosos: {len(modelos_entrenados)}")
    print(f"  Modelos fallidos: {modelos_fallidos}")
    
    # Añadimos la fecha de entrenamiento a los datos guardados
    datos_a_guardar = {
        'fecha_entrenamiento': datetime.now(),
        'modelos': modelos_entrenados
    }
    
    print(f"Guardando modelos en '{ARCHIVO_MODELOS_SALIDA}'...")
    joblib.dump(datos_a_guardar, ARCHIVO_MODELOS_SALIDA)
    
    end_time = datetime.now()
    print("\n--- PROCESO FINALIZADO ---")
    print(f"Tiempo total: {end_time - start_time}")

if __name__ == "__main__":
    main()