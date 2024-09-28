import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------

# Dataframes del sujeto 1
df1 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T2.parquet', engine='pyarrow')
df_s1 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
# Dataframes del sujeto 2
df1 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T2.parquet', engine='pyarrow')
df_s2 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Creamos un solo dataframe y eliminamos columnas 9 y 10
df = pd.concat([df_s1, df_s2], ignore_index=True)
df = df.drop(columns=[df.columns[8], df.columns[9]])
# Para saber cuántos datos de los estímulos hay.
# Conteo_por_estado = df[df.columns[-1]].value_counts()

# Se añade la columna bloque de modo que se cuente el cambio de estímulo.
c_estimulo = df.columns[-1]
df['bloque'] = (df[c_estimulo] != df[c_estimulo].shift()).cumsum()

#---------------------------------------------------------------------------------

# Visualización específica. Se ven las señales de los 7 agarres de un solo sensor.
fig, axs = plt.subplots(4, 2, figsize=(30, 20))
# Iterar sobre los 7 tipos de agarre
for i in range(7):
    # Definir el rango de bloques para cada agarre
    bloque_inicio = i * 24 + 1
    bloque_fin = (i + 1) * 24
    # Filtrar los datos del rango de bloques correspondiente
    df_filtrado = df[df['bloque'].between(bloque_inicio, bloque_fin)]
    # Seleccionar los valores de la primera columna (primer sensor)
    valores_sensor_1 = df_filtrado.iloc[:, 0]
    # Determinar la posición en la matriz de subplots
    columna = i // 4
    fila = i % 4
    # Graficar en el subplot correspondiente
    axs[fila, columna].plot(valores_sensor_1.index, valores_sensor_1)
    axs[fila, columna].set_title(f'Agarre {i + 1}')
    axs[fila, columna].set_xlabel('Muestras')
    axs[fila, columna].set_ylabel('Amplitud de señal EMG')
# Ajustar el espacio entre las gráficas
plt.tight_layout()
# Mostrar la figura
plt.show()

#---------------------------------------------------------------------------------

# Definimos el número de sensores (son 14)
num_sensores = 14
# Creamos una lista para almacenar las filas del DataFrame global
filas_df_global = []
# Definimos el número total de bloques
num_bloques = df['bloque'].max()

# Iterar sobre los intervalos de 24 bloques
for i in range(0, num_bloques, 24):
    # Creamos una lista para almacenar los DataFrames de una fila
    fila = []
    # Filtramos los datos de los bloques correspondientes al intervalo actual
    df_filtrado = df[df['bloque'].between(i + 1, i + 24)]
    # Iteramos sobre cada sensor (las primeras 14 columnas)
    for sensor in range(num_sensores):
        # Crear un DataFrame para los valores del sensor en este rango de bloques
        df_sensor = df_filtrado.iloc[:, sensor]
        # Añadir este DataFrame a la fila
        fila.append(df_sensor)
    # Obtener el valor del estímulo (asumimos que todos los valores en el rango de bloques son iguales)
    estimulo = df_filtrado['stimulus'].iloc[0]
    # Añadir la fila con los DataFrames de los sensores y el estímulo
    fila.append(estimulo)
    # Añadir esta fila al DataFrame global
    filas_df_global.append(fila)

# Crear el DataFrame global con las filas de DataFrames de sensores y la columna del estímulo
columnas = [f'Sensor_{i+1}' for i in range(num_sensores)] + ['Stimulus']
df_global = pd.DataFrame(filas_df_global, columns=columnas)
# Es importante mencionar que esta df_global tiene como elementos un conjunto de datos.
# Y aunque pueden graficar las señales, se ha perdido la información sobre los "0", es decir el no agarre.
# Por lo tanto, un elemento contiene 12 repeticiones de un tipo de agarre de un tipo de sensor.
# Pero ya no se indica, dentro de esas repeticiones, cuándo hay agarre y cuándo no.
# Este dataframe puede servir si solo interesa cada "serie de repeticiones" y no cuando hay agarre.
# Pueden extraerse datos con, por ejemplo, df_global.iloc[0, 0] (fila 1, columna 1)
# En este df, por ejemplo, pueden sacarse características por cada elemento, ya que son señales completas.

#---------------------------------------------------------------------------------



