import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import holidays  # <-- ¡NUEVA IMPORTACIÓN!

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constantes ---
ARCHIVO_DATOS = 'ventas_farmacia_fake.csv'
ARCHIVO_MODELOS_SALIDA = 'modelos_farmacia.joblib'
ARCHIVO_CLIMA = 'clima_madrid.csv'

# --- Funciones ---

def cargar_datos(file_name):
    """Carga y prepara los datos iniciales desde el CSV."""
    print(f"Cargando datos de '{file_name}'...")
    df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
    return df

def cargar_clima(file_name):
    """Carga los datos climáticos descargados."""
    try:
        df_clima = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha']).dt.date
        print("Datos climáticos cargados exitosamente.")
        return df_clima
    except FileNotFoundError:
        print(f"Advertencia: No se encontró '{file_name}'. El modelo se entrenará sin datos climáticos.")
        return None

def crear_features_avanzadas(df_in, df_clima):
    """
    Crea características de tiempo, lag, rolling y externas (clima, festivos).
    """
    df = df_in.copy()
    
    # 1. Características de tiempo base
    df['mes'] = df['ds'].dt.month
    df['dia_del_ano'] = df['ds'].dt.dayofyear
    df['dia_de_la_semana'] = df['ds'].dt.dayofweek
    df['ano'] = df['ds'].dt.year
    
    # 2. Características Externas (Reales y Simuladas Inteligentes)
    
    # Festivos Reales (España)
    festivos_espana = holidays.Spain(years=[2022, 2023, 2024])
    df['es_festivo'] = df['ds'].isin(festivos_espana).astype(int)
    
    # Temporadas (Conocimiento del Dominio)
    # Temporada de Gripe (Oct-Mar)
    df['temporada_gripe'] = df['mes'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    # Temporada de Polen (Mar-Jun)
    df['temporada_polen'] = df['mes'].isin([3, 4, 5, 6]).astype(int)
    
    # Datos Climáticos Reales (Temperatura)
    if df_clima is not None:
        df_clima['ds'] = pd.to_datetime(df_clima['Fecha'])
        df = pd.merge(df, df_clima[['ds', 'Temperatura_Media']], on='ds', how='left')
        df['Temperatura_Media'] = df['Temperatura_Media'].fillna(method='ffill').fillna(method='bfill') # Rellenar huecos
    else:
        # Si el archivo de clima falla, crea una columna "dummy" para que el modelo no falle
        df['Temperatura_Media'] = 15.0 # Valor neutro
    
    # 3. Características de Retraso (Lag)
    df['ventas_lag_1'] = df['y'].shift(1)
    df['ventas_lag_7'] = df['y'].shift(7)
    
    # 4. Características Móviles (Rolling)
    df['media_movil_7d'] = df['y'].shift(1).rolling(window=7, min_periods=1).mean()
    df['media_movil_30d'] = df['y'].shift(1).rolling(window=30, min_periods=1).mean()
    
    # Rellenar NaNs al principio de la serie
    df = df.bfill()
    
    return df

def entrenar_modelo_para_serie(df_serie, df_clima):
    """
    Entrena un único modelo XGBoost para una serie de tiempo.
    """
    df_diario = df_serie.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    df_diario = df_diario.set_index('ds').asfreq('D').fillna(0).reset_index()

    if df_diario.empty or len(df_diario) < 100: 
        return None, None, None

    # 1. Crear todas las features avanzadas
    df_preparado = crear_features_avanzadas(df_diario, df_clima)
    
    # 2. Definir las nuevas features
    features = [
        'mes', 'dia_del_ano', 'dia_de_la_semana', 'ano', 'es_festivo',
        'temporada_gripe', 'temporada_polen', 'Temperatura_Media',
        'ventas_lag_1', 'ventas_lag_7',
        'media_movil_7d', 'media_movil_30d'
    ]
    target = 'y'

    # Eliminar filas con NaNs
    df_preparado = df_preparado.dropna(subset=features)
    
    if df_preparado.empty:
        return None, None, None
        
    X = df_preparado[features]
    y = df_preparado[target]

    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=50, random_state=42,
        eval_metric='rmse'
    )

    eval_size = int(len(X) * 0.2)
    rmse = None
    
    if eval_size < 50: 
        model.fit(X, y, verbose=False)
        try:
            y_pred = model.predict(X)
            rmse = sqrt(mean_squared_error(y, y_pred))
        except:
            rmse = -1
    else:
        X_train, X_eval = X.iloc[:-eval_size], X.iloc[-eval_size:]
        y_train, y_eval = y.iloc[:-eval_size], y.iloc[-eval_size:]
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        rmse = model.best_score
    
    df_importancia = pd.DataFrame({
        'Impulsor': features,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    return model, rmse, df_importancia

# --- BLOQUE PRINCIPAL DE ENTRENAMIENTO ---
def main():
    print("--- INICIANDO ENTRENAMIENTO AVANZADO (Datos Externos Reales) ---")
    start_time = datetime.now()
    
    df_total = cargar_datos(ARCHIVO_DATOS)
    df_clima = cargar_clima(ARCHIVO_CLIMA)
    
    combinaciones = df_total[['Farmacia_ID', 'Producto']].drop_duplicates()
    
    print(f"Se entrenarán modelos para {len(combinaciones)} combinaciones...")
    
    modelos_entrenados = {}
    modelos_fallidos = 0
    
    for idx, row in combinaciones.iterrows():
        farmacia_id = row['Farmacia_ID']
        producto = row['Producto']
        clave_modelo = f"{farmacia_id}::{producto}" 
        print(f"  Entrenando: {clave_modelo} ({idx+1}/{len(combinaciones)})")
        
        df_serie = df_total[
            (df_total['Farmacia_ID'] == farmacia_id) &
            (df_total['Producto'] == producto)
        ]
        
        # Pasamos los datos climáticos a la función de entrenamiento
        modelo, rmse, df_importancia = entrenar_modelo_para_serie(df_serie, df_clima)
        
        if modelo and rmse is not None and df_importancia is not None:
            modelos_entrenados[clave_modelo] = {
                'model': modelo,
                'rmse': rmse,
                'importance': df_importancia
            }
            print(f"    -> Éxito. RMSE: {rmse:.2f}")
        else:
            print(f"    -> Fallido (datos insuficientes)")
            modelos_fallidos += 1
            
    print("\nEntrenamiento completado.")
    print(f"  Modelos exitosos: {len(modelos_entrenados)}")
    print(f"  Modelos fallidos: {modelos_fallidos}")
    
    datos_a_guardar = {
        'fecha_entrenamiento': datetime.now(),
        'modelos': modelos_entrenados
    }
    
    print(f"Guardando modelos, métricas e importancia en '{ARCHIVO_MODELOS_SALIDA}'...")
    joblib.dump(datos_a_guardar, ARCHIVO_MODELOS_SALIDA)
    
    end_time = datetime.now()
    print("\n--- PROCESO FINALIZADO ---")
    print(f"Tiempo total: {end_time - start_time}")

if __name__ == "__main__":
    main()