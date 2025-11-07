import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def descargar_y_guardar_clima():
    """
    Se conecta a la API de Open-Meteo para descargar datos históricos
    de temperatura para Madrid (2022-2024) y los guarda en un CSV.
    """
    print("Iniciando descarga de datos climáticos históricos (Solo Temperatura)...")
    
    # --- Configuración de la API ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Coordenadas de Madrid y rango de fechas de nuestros datos
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.4165,
        "longitude": -3.7026,
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "daily": ["temperature_2m_mean"], # <-- CORREGIDO: Solo pedimos temperatura
        "timezone": "Europe/Berlin"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        print(f"Datos recibidos de Open-Meteo para: {response.Latitude()}°N {response.Longitude()}°E")

        # --- Procesar Datos Diarios (Temperatura) ---
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
        
        # --- Limpiar y Guardar ---
        df_daily['Fecha'] = pd.to_datetime(df_daily['date'].dt.date)
        df_clima_final = df_daily[['Fecha', 'Temperatura_Media']]
        df_clima_final['Fecha'] = pd.to_datetime(df_clima_final['Fecha']).dt.date
        
        # Guardar en CSV
        output_file = "clima_madrid.csv"
        df_clima_final.to_csv(output_file, index=False, sep=';', decimal=',')
        
        print(f"¡Éxito! Datos climáticos (Temperatura) guardados en '{output_file}'.")
        print(df_clima_final.head())

    except Exception as e:
        print(f"Error al descargar o procesar los datos climáticos: {e}")
        print("Asegúrate de tener conexión a internet y las librerías 'openmeteo-requests', 'requests_cache' y 'retry_requests' instaladas.")

if __name__ == "__main__":
    descargar_y_guardar_clima()