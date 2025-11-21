import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import holidays
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Ignorar advertencias
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- Constantes ---
ARCHIVO_DATOS = 'ventas_farmacia_fake.csv'
ARCHIVO_MODELOS_SALIDA = 'modelos_farmacia.joblib'
ARCHIVO_CLIMA = 'clima_madrid.csv'

# --- Funciones ---

def cargar_datos(file_name):
    print(f"⏳ Cargando ventas desde '{file_name}'...")
    df = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
    return df

def cargar_clima(file_name):
    try:
        df_clima = pd.read_csv(file_name, delimiter=';', decimal=',', parse_dates=['Fecha'])
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha']).dt.date
        print(f"✅ Datos climáticos cargados ({len(df_clima)} registros).")
        return df_clima
    except FileNotFoundError:
        print(f"⚠️ ADVERTENCIA: No se encontró '{file_name}'.")
        return None

def crear_features_premium(df_in, df_clima):
    df = df_in.copy()
    df['mes'] = df['ds'].dt.month
    df['dia_semana'] = df['ds'].dt.dayofyear
    df['dia_ano'] = df['ds'].dt.dayofyear
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek / 7)

    years = df['ds'].dt.year.unique()
    es_holidays = holidays.Spain(years=years)
    df['es_festivo'] = df['ds'].isin(es_holidays).astype(int)
    
    df['temp_gripe'] = df['mes'].isin([10, 11, 12, 1, 2]).astype(int)
    df['temp_alergia'] = df['mes'].isin([3, 4, 5, 6]).astype(int)

    if df_clima is not None:
        df_clima['ds'] = pd.to_datetime(df_clima['Fecha'])
        df = pd.merge(df, df_clima[['ds', 'Temperatura_Media']], on='ds', how='left')
        df['Temperatura_Media'] = df['Temperatura_Media'].interpolate(method='linear').bfill().ffill()
    else:
        df['Temperatura_Media'] = 15.0

    df['lag_1'] = df['y'].shift(1)
    df['lag_2'] = df['y'].shift(2)
    df['lag_7'] = df['y'].shift(7)
    df['lag_14'] = df['y'].shift(14)

    df['roll_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
    df['roll_mean_28'] = df['y'].shift(1).rolling(window=28).mean()
    df['roll_std_7'] = df['y'].shift(1).rolling(window=7).std()
    df['tendencia_semanal'] = df['roll_mean_7'] - df['roll_mean_7'].shift(7)

    df = df.dropna()
    return df

def entrenar_y_optimizar(df_serie, df_clima):
    df_diario = df_serie.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    df_diario = df_diario.set_index('ds').asfreq('D').fillna(0).reset_index()

    if len(df_diario) < 90: return None, None, None

    df_features = crear_features_premium(df_diario, df_clima)
    if df_features.empty: return None, None, None

    features_cols = [
        'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 
        'es_festivo', 'temp_gripe', 'temp_alergia', 'Temperatura_Media',
        'lag_1', 'lag_2', 'lag_7', 'lag_14',
        'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'tendencia_semanal'
    ]
    target = 'y'
    X = df_features[features_cols]
    y = df_features[target]

    test_size = int(len(X) * 0.15)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # --- AJUSTE PARA REDUCIR TAMAÑO ---
    # Reducimos el número de estimadores para que el archivo final sea más ligero
    param_grid = {
        'n_estimators': [300, 500, 700],     # Reducido de [500, 1000, 1500]
        'learning_rate': [0.01, 0.05],       
        'max_depth': [3, 5],                 # Reducido profundidad máxima
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    model_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=model_base,
        param_distributions=param_grid,
        n_iter=5, # Menos iteraciones para ir más rápido
        scoring='neg_root_mean_squared_error',
        cv=TimeSeriesSplit(n_splits=3),
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = np.maximum(best_model.predict(X_test), 0)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    df_importancia = pd.DataFrame({
        'Impulsor': features_cols,
        'Importancia': best_model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    return best_model, rmse, df_importancia

def main():
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO (LIGERO)")
    start_time = datetime.now()
    
    df_total = cargar_datos(ARCHIVO_DATOS)
    df_clima = cargar_clima(ARCHIVO_CLIMA)
    
    combinaciones = df_total[['Farmacia_ID', 'Producto']].drop_duplicates()
    total_comb = len(combinaciones)
    
    modelos_entrenados = {}
    
    for idx, row in combinaciones.iterrows():
        farmacia_id = row['Farmacia_ID']
        producto = row['Producto']
        clave_modelo = f"{farmacia_id}::{producto}"
        
        print(f"[{idx+1}/{total_comb}] {producto} ({farmacia_id})...", end=" ", flush=True)
        
        df_serie = df_total[
            (df_total['Farmacia_ID'] == farmacia_id) &
            (df_total['Producto'] == producto)
        ]
        
        modelo, rmse, df_importancia = entrenar_y_optimizar(df_serie, df_clima)
        
        if modelo:
            modelos_entrenados[clave_modelo] = {
                'model': modelo,
                'rmse': rmse,
                'importance': df_importancia
            }
            print(f"✅ RMSE: {rmse:.2f}")
        else:
            print(f"❌")
            
    print("\n✨ Entrenamiento completado.")
    
    datos_a_guardar = {
        'fecha_entrenamiento': datetime.now(),
        'modelos': modelos_entrenados
    }
    
    print(f"💾 Comprimiendo y guardando en '{ARCHIVO_MODELOS_SALIDA}'...")
    # --- ¡LA CLAVE ESTÁ AQUÍ! compress=3 reduce el tamaño enormemente ---
    joblib.dump(datos_a_guardar, ARCHIVO_MODELOS_SALIDA, compress=3)
    
    duracion = datetime.now() - start_time
    print(f"🏁 Tiempo total: {duracion}")

if __name__ == "__main__":
    main()