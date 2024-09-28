import pandas as pd
import matplotlib.pyplot as plt

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

# Visualización específica. Se ve
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