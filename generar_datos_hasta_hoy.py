import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuración
ARCHIVO_ORIGINAL = 'ventas_farmacia_fake.csv'
ARCHIVO_ACTUALIZADO = 'ventas_farmacia_fake.csv' # Sobreescribiremos el mismo (o cambia el nombre si prefieres backup)

def generar_datos_actualizados():
    print(f"--- CARGANDO DATOS ORIGINALES ---")
    try:
        df = pd.read_csv(ARCHIVO_ORIGINAL, delimiter=';', decimal=',')
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        print(f"Filas originales: {len(df)}")
    except FileNotFoundError:
        print("Error: No se encuentra el archivo original.")
        return

    # 1. Detectar la última fecha registrada
    ultima_fecha = df['Fecha'].max()
    fecha_hoy = datetime.now()
    
    print(f"Última fecha en datos: {ultima_fecha.date()}")
    print(f"Fecha de hoy: {fecha_hoy.date()}")
    
    if ultima_fecha >= fecha_hoy:
        print("¡Los datos ya están actualizados! No es necesario generar nada.")
        return

    # 2. Crear un "Catálogo Maestro" de productos y farmacias
    # Esto asegura que usamos los mismos precios, categorías y zonas que ya existen
    catalogo = df[['Farmacia_ID', 'Zona_Farmacia', 'Producto', 'Categoria', 'Precio_Unitario_€']].drop_duplicates()
    
    # Calculamos la venta media diaria por producto para que la simulación sea realista
    # (Media de cantidad cuando se vende)
    medias_venta = df.groupby('Producto')['Cantidad'].mean().to_dict()
    
    nuevas_filas = []
    
    # 3. Bucle día a día desde la última fecha hasta hoy
    dias_a_generar = (fecha_hoy - ultima_fecha).days
    print(f"Generando datos para {dias_a_generar} días faltantes...")
    
    fecha_actual = ultima_fecha + timedelta(days=1)
    
    while fecha_actual <= fecha_hoy:
        
        # Para cada día, simulamos ventas en cada farmacia
        for _, row in catalogo.iterrows():
            
            # Lógica de Estacionalidad Simple
            mes = fecha_actual.month
            es_invierno = mes in [10, 11, 12, 1, 2]
            es_primavera = mes in [3, 4, 5, 6]
            
            probabilidad_venta = 0.7 # 70% de probabilidad de que se venda el producto ese día
            multiplicador_cantidad = 1.0
            
            # Ajustar probabilidad y cantidad según la época
            if row['Categoria'] == 'Antigripal' and es_invierno:
                probabilidad_venta = 0.95
                multiplicador_cantidad = 1.5
            elif row['Categoria'] == 'Antigripal' and not es_invierno:
                probabilidad_venta = 0.3
                multiplicador_cantidad = 0.5
                
            if row['Categoria'] == 'Alergia' and es_primavera:
                probabilidad_venta = 0.90
                multiplicador_cantidad = 1.4
            elif row['Categoria'] == 'Alergia' and not es_primavera:
                probabilidad_venta = 0.2
                multiplicador_cantidad = 0.5

            # Decidir si hay venta
            if random.random() < probabilidad_venta:
                # Generar cantidad basada en la media histórica + ruido aleatorio
                media_prod = medias_venta.get(row['Producto'], 10)
                cantidad_base = int(np.random.normal(media_prod, media_prod * 0.3)) # Variación del 30%
                cantidad = int(max(1, cantidad_base * multiplicador_cantidad)) # Nunca menos de 1
                
                total_venta = round(cantidad * row['Precio_Unitario_€'], 2)
                
                nuevas_filas.append({
                    'Fecha': fecha_actual,
                    'Farmacia_ID': row['Farmacia_ID'],
                    'Zona_Farmacia': row['Zona_Farmacia'],
                    'Producto': row['Producto'],
                    'Categoria': row['Categoria'],
                    'Cantidad': cantidad,
                    'Precio_Unitario_€': row['Precio_Unitario_€'],
                    'Total_Venta_€': total_venta
                })
        
        fecha_actual += timedelta(days=1)

    # 4. Unir y Guardar
    if nuevas_filas:
        df_nuevo = pd.DataFrame(nuevas_filas)
        print(f"Se han generado {len(df_nuevo)} nuevas líneas de venta.")
        
        df_completo = pd.concat([df, df_nuevo])
        
        # Ordenar por fecha
        df_completo = df_completo.sort_values('Fecha')
        
        # Guardar (usando el formato correcto para tu CSV: ; y , decimal)
        df_completo.to_csv(
            ARCHIVO_ACTUALIZADO, 
            index=False, 
            sep=';', 
            decimal=',',
            date_format='%Y-%m-%d'
        )
        print(f"¡ÉXITO! Archivo '{ARCHIVO_ACTUALIZADO}' actualizado hasta {fecha_hoy.date()}.")
        
        # Pequeña validación
        print(f"Fecha final en el nuevo archivo: {df_completo['Fecha'].max()}")
        
    else:
        print("No se generaron filas nuevas.")

if __name__ == "__main__":
    generar_datos_actualizados()