from tqdm import tqdm
import filters_and_features as ff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

#---------------------------------------------------------------------------------

def lectura(nivel_ram=3, ruta_salida='../dataframes/', nombre_archivo='df.parquet'):
    # Ruta base donde se encuentran los archivos
    base_path = '../datasets/'
    
    # Crear el directorio de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    # Segmentaciones según el nivel de RAM
    segmentaciones = {
        1: 5,    # Procesar archivos de S en bloques pequeños (nivel bajo de RAM)
        2: 10,   # Procesar archivos de S en bloques moderados
        3: 15,   # Procesar archivos de S en bloques más grandes
        4: 20,   # Procesar archivos de S en bloques grandes
        5: 50    # Cargar una cantidad mayor de archivos en cada bloque
    }

    # Tamaño del bloque de archivos a leer según nivel de RAM
    tamano_bloque = segmentaciones.get(nivel_ram, 10)

    # Contar cuántos archivos totales hay para mostrar barra de progreso
    total_files = sum([1 for i in range(1, 11) for j in range(1, 6) for k in range(1, 3)])

    # Barra de progreso
    progress = tqdm(total=total_files, desc="Procesando archivos Parquet")

    # Crear un ParquetWriter para ir agregando tablas de datos
    parquet_path = f'{ruta_salida}{nombre_archivo}'
    parquet_writer = None

    # Variable para rastrear el número acumulado de bloque
    bloque_acumulado = 0

    # Iterar en bloques sobre las combinaciones de S, D y T
    for i in range(1, 11, tamano_bloque):  # Procesar de S en bloques
        for s in range(i, min(i + tamano_bloque, 11)):  # Bloque de archivos
            dfs = []  # Lista para almacenar temporalmente los DataFrames del bloque
            for j in range(1, 6):  # D del 1 al 5
                for k in range(1, 3):  # T del 1 al 2
                    file_path = f'{base_path}S{s}_D{j}_T{k}.parquet'
                    try:
                        # Leer el archivo Parquet
                        df = pd.read_parquet(file_path, engine='pyarrow')

                        # Añadir la columna de bloques (como en tu función original)
                        c_estimulo = df.columns[-1]
                        df['bloque'] = (df[c_estimulo] != df[c_estimulo].shift()).cumsum()

                        # Ajustar el valor del bloque para continuar con el valor acumulado
                        df['bloque'] += bloque_acumulado

                        # Actualizar el bloque acumulado al último valor de 'bloque' del DataFrame actual
                        bloque_acumulado = df['bloque'].iloc[-1]

                        # Agregar a la lista de DataFrames
                        dfs.append(df)
                        
                    except FileNotFoundError:
                        # Si el archivo no existe, continuar con el siguiente
                        continue

                    # Actualizar la barra de progreso
                    progress.update(1)

            # Concatenar el bloque de DataFrames
            if dfs:
                df_bloque = pd.concat(dfs, ignore_index=True)
                table = pa.Table.from_pandas(df_bloque)

                # Inicializar el ParquetWriter si aún no existe
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')

                # Escribir el bloque en el archivo Parquet
                parquet_writer.write_table(table)

                # Limpiar memoria después de cada bloque
                del dfs, df_bloque, table, df  # Eliminar explícitamente todas las variables del bloque
                gc.collect()  # Forzar recolección de basura

    # Cerrar el ParquetWriter cuando todo esté escrito
    if parquet_writer:
        parquet_writer.close()

    # Cerrar la barra de progreso
    progress.close()

#---------------------------------------------------------------------------------

