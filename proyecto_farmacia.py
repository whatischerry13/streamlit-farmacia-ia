import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import warnings

# Ignorar advertencias futuras de pandas/altair
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURACIÓN ---
ARCHIVO_DATOS = 'ventas_farmacia_fake.csv'
PRODUCTO_OBJETIVO = 'Paracetamol 500mg'
DIAS_A_PRONOSTICAR = 90


def cargar_datos(file_name):
    """Carga y prepara los datos iniciales desde el CSV."""
    print(f"--- FASE 1: Cargando datos de '{file_name}' ---")
    try:
        df = pd.read_csv(
            file_name,
            delimiter=';',
            decimal=',',
            parse_dates=['Fecha']
        )
        print("Datos cargados exitosamente.")
        print(f"Rango de fechas: {df['Fecha'].min()} a {df['Fecha'].max()}")
        print(f"{len(df)} filas cargadas.\n")
        return df
    except FileNotFoundError:
        print(f"¡ERROR! No se encontró el archivo '{file_name}'.")
        print("Asegúrate de que el archivo CSV está en la misma carpeta que el script.")
        return None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def analizar_epocas(df):
    """
    (FASE 3 - CORREGIDA) 
    Genera el gráfico de épocas Alergia vs. Antigripal.
    """
    print("--- FASE 3: Analizando 'Épocas' (Alergia vs. Antigripal) ---")
    
    categorias_interes = ['Alergia', 'Antigripal']
    df_filtrado = df[df['Categoria'].isin(categorias_interes)].copy()

    # --- INICIO DE LA CORRECCIÓN ---
    
    # 1. Establecemos 'Fecha' como el índice del DataFrame
    #    Esto hace que el resampleo (paso 2) sea mucho más estable.
    df_filtrado.set_index('Fecha', inplace=True)

    # 2. Ahora agrupamos por 'Categoria' y remuestreamos ('W' - semanal)
    #    Ya no necesitamos 'on="Fecha"' porque resample() usará el índice.
    df_semanal = df_filtrado.groupby('Categoria').resample('W')['Cantidad'].sum()
    
    # 3. Reseteamos el índice para que 'Fecha' y 'Categoria' vuelvan a ser columnas
    #    (necesario para el gráfico de Altair)
    df_semanal = df_semanal.reset_index()

    # --- FIN DE LA CORRECIÓN ---
    
    # Crear el gráfico (esta parte no cambia)
    chart = alt.Chart(df_semanal).mark_line(point=True).encode(
        x=alt.X('Fecha', title='Fecha (Semanas)'),
        y=alt.Y('Cantidad', title='Cantidad Total Vendida'),
        color=alt.Color('Categoria', title='Categoría'),
        tooltip=['Fecha', 'Categoria', 'Cantidad']
    ).properties(
        title='Ventas Semanales: Épocas de Alergia vs. Antigripal',
        width=800
    ).interactive()

    # Guardar como HTML para ver en el navegador
    output_filename = 'grafico_epocas_alergia_antigripal.html'
    chart.save(output_filename)
    print(f"Gráfico de Épocas guardado en: '{output_filename}'\n")


def crear_caracteristicas_temporales(df_in):
    """(FASE 2) Crea características de tiempo para el modelo de ML."""
    df_out = df_in.copy()
    df_out['mes'] = df_out['ds'].dt.month
    df_out['dia_del_ano'] = df_out['ds'].dt.dayofyear
    df_out['semana_del_ano'] = df_out['ds'].dt.isocalendar().week.astype(int)
    df_out['dia_de_la_semana'] = df_out['ds'].dt.dayofweek
    df_out['ano'] = df_out['ds'].dt.year
    df_out['trimestre'] = df_out['ds'].dt.quarter
    return df_out

def pronosticar_stock_xgboost(df, producto, dias_futuros):
    """(FASE 4) Entrena un modelo XGBoost y genera un pronóstico."""
    print(f"--- FASE 4: Modelado de Stock (XGBoost) para '{producto}' ---")

    # 1. Preparar datos
    df_producto = df[df['Producto'] == producto].copy()
    df_diario = df_producto.groupby('Fecha')['Cantidad'].sum().reset_index()
    df_diario = df_diario.rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    
    print(f"Datos de '{producto}' agregados por día.")

    # 2. Ingeniería de Características
    df_preparado = crear_caracteristicas_temporales(df_diario)
    
    # 3. Definir Características (X) y Objetivo (y)
    features = ['mes', 'dia_del_ano', 'semana_del_ano', 'dia_de_la_semana', 'ano', 'trimestre']
    target = 'y'

    X = df_preparado[features]
    y = df_preparado[target]

    # 4. Entrenar el Modelo XGBoost
    print("Entrenando modelo XGBoost...")
    
    # Dividir para validación interna de XGBoost (early stopping)
    eval_size = int(len(X) * 0.2) # Usar 20% para validación
    X_train, X_eval = X.iloc[:-eval_size], X.iloc[-eval_size:]
    y_train, y_eval = y.iloc[:-eval_size], y.iloc[-eval_size:]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,      # Número de árboles (alto, pero con early stopping)
        learning_rate=0.01,     # Tasa de aprendizaje (pequeña)
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50, # Parará si no mejora en 50 rondas
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False # Poner True si quieres ver el proceso de entrenamiento
    )
    
    print("Modelo entrenado exitosamente.")

    # 5. Generar Pronóstico
    print(f"Generando pronóstico a {dias_futuros} días...")
    
    fecha_maxima_historica = df_preparado['ds'].max()
    fechas_futuras = pd.date_range(
        start=fecha_maxima_historica + pd.Timedelta(days=1),
        periods=dias_futuros,
        freq='D'
    )
    
    df_futuro = pd.DataFrame({'ds': fechas_futuras})
    df_futuro_preparado = crear_caracteristicas_temporales(df_futuro)
    
    prediccion_futura = model.predict(df_futuro_preparado[features])
    prediccion_futura[prediccion_futura < 0] = 0 # No podemos predecir ventas negativas
    
    df_futuro['Prediccion'] = prediccion_futura.round(0).astype(int)

    # 6. Guardar Pronóstico (CSV)
    output_csv = f'pronostico_{producto.replace(" ", "_").lower()}_{dias_futuros}_dias.csv'
    df_futuro_final = df_futuro.rename(columns={'ds': 'Fecha', 'Prediccion': 'Prediccion_Cantidad'})
    df_futuro_final.to_csv(output_csv, index=False, sep=';', decimal=',')
    print(f"Datos del pronóstico guardados en: '{output_csv}'")

    # 7. Visualizar Pronóstico (HTML)
    df_plot_real = df_preparado[['ds', 'y']].rename(columns={'y': 'Ventas'})
    df_plot_real['Tipo'] = 'Real'
    
    df_plot_pred = df_futuro[['ds', 'Prediccion']].rename(columns={'Prediccion': 'Ventas'})
    df_plot_pred['Tipo'] = 'Predicción'
    
    # Graficar solo el último año real + predicción
    df_plot_real_reciente = df_plot_real[df_plot_real['ds'] > (fecha_maxima_historica - pd.Timedelta(days=365))]
    df_plot_combinado = pd.concat([df_plot_real_reciente, df_plot_pred])

    chart = alt.Chart(df_plot_combinado).mark_line().encode(
        x=alt.X('ds', title='Fecha'),
        y=alt.Y('Ventas', title='Cantidad Vendida'),
        color=alt.Color('Tipo', title='Dato'),
        strokeDash=alt.StrokeDash('Tipo', title='Dato'),
        tooltip=['ds', 'Ventas', 'Tipo']
    ).properties(
        title=f'Pronóstico XGBoost: {producto} (Último Año + {dias_futuros} Días)',
        width=800
    ).interactive()
    
    output_html = f'grafico_pronostico_{producto.replace(" ", "_").lower()}.html'
    chart.save(output_html)
    print(f"Gráfico de pronóstico guardado en: '{output_html}'\n")

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
def main():
    df = cargar_datos(ARCHIVO_DATOS)
    
    if df is not None:
        # 1. Analizar las épocas generales (FUNCIÓN CORREGIDA)
        analizar_epocas(df)
        
        # 2. Crear el pronóstico de stock para el producto estrella
        pronosticar_stock_xgboost(df, PRODUCTO_OBJETIVO, DIAS_A_PRONOSTICAR)
        
        print("--- Proceso completado ---")
        print("Revisa los archivos .html y .csv generados en la carpeta del proyecto.")

if __name__ == "__main__":
    main()