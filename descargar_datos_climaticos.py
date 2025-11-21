import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

def descargar_y_guardar_clima():
    """
    Descarga datos climáticos de Madrid desde 2022 hasta AYER.
    """
    print("Iniciando descarga de datos climáticos actualizados...")
    
    # --- 1. Calcular Fechas Dinámicas ---
    fecha_inicio = "2022-01-01"
    # La API de archivo suele tener un retraso de 2-3 días. Pedimos hasta hace 2 días.
    fecha_fin = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    
    print(f"Solicitando datos desde {fecha_inicio} hasta {fecha_fin}...")

    # --- 2. Configuración API ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.4165,
        "longitude": -3.7026,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "daily": ["temperature_2m_mean"],
        "timezone": "Europe/Berlin"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # --- 3. Procesar Datos ---
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}
        daily_data["Temperatura_Media"] = daily_temperature_2m_mean
        df_daily = pd.DataFrame(data=daily_data)
        
        # Limpieza y formato
        df_daily['Fecha'] = pd.to_datetime(df_daily['date'].dt.date)
        df_clima_final = df_daily[['Fecha', 'Temperatura_Media']]
        
        # --- 4. Guardar ---
        output_file = "clima_madrid.csv"
        df_clima_final.to_csv(output_file, index=False, sep=';', decimal=',')
        
        print(f"¡ÉXITO! Datos climáticos actualizados guardados en '{output_file}'.")
        print(f"Rango: {df_clima_final['Fecha'].min().date()} a {df_clima_final['Fecha'].max().date()}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    descargar_y_guardar_clima()