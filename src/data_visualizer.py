import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------
def lectura():
    # Dataframes del sujeto 1
    df1 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D1_T1.parquet', engine='pyarrow')
    df2 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D1_T2.parquet', engine='pyarrow')
    df3 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D2_T1.parquet', engine='pyarrow')
    df4 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D2_T2.parquet', engine='pyarrow')
    df5 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D3_T1.parquet', engine='pyarrow')
    df6 = pd.read_parquet('../datasets/ninapro/DB6_s1_a/S1_D3_T2.parquet', engine='pyarrow')
    df_s1 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    # Dataframes del sujeto 2
    df1 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D1_T1.parquet', engine='pyarrow')
    df2 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D1_T2.parquet', engine='pyarrow')
    df3 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D2_T1.parquet', engine='pyarrow')
    df4 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D2_T2.parquet', engine='pyarrow')
    df5 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D3_T1.parquet', engine='pyarrow')
    df6 = pd.read_parquet('../datasets/ninapro/DB6_s2_a/S2_D3_T2.parquet', engine='pyarrow')
    df_s2 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    
    # Creamos un solo dataframe y eliminamos columnas 9 y 10
    df = pd.concat([df_s1, df_s2], ignore_index=True)
    df = df.drop(columns=[df.columns[8], df.columns[9]])
    # Para saber cuántos datos de los estímulos hay.
    # Conteo_por_estado = df[df.columns[-1]].value_counts()
    
    # Se añade la columna bloque de modo que se cuente el cambio de estímulo.
    c_estimulo = df.columns[-1]
    df['bloque'] = (df[c_estimulo] != df[c_estimulo].shift()).cumsum()
    # Filtrar las filas donde los bloques no estén en el grupo 62
    inicio_grupo_62 = (64 - 1) * 24 + 1  # Primer bloque del grupo 63
    fin_grupo_62 = 64 * 24               # Último bloque del grupo 63
    df = df[~df['bloque'].between(inicio_grupo_62, fin_grupo_62)]
    # Se eliminan las columnas de bloques existentes
    df = df.drop(columns=['bloque'])
    
    # Se vuelve a crear
    c_estimulo = df.columns[-1]
    df['bloque'] = (df[c_estimulo] != df[c_estimulo].shift()).cumsum()
    
    return df

#---------------------------------------------------------------------------------

def raw_viewer(df,n_col=0,k=1):
    # Visualización específica. Se ven las señales de los 7 agarres de un solo sensor.
    fig, axs = plt.subplots(4, 2, figsize=(30, 20))
    # Iterar sobre los 7 tipos de agarre
    for i in range(7):
        # Definir el rango de bloques para cada agarre
        bloque_inicio = i * 24 + 1
        bloque_fin = (i + 1) * 24
        # Filtrar los datos del rango de bloques correspondiente
        df_filtrado = df[df['bloque'].between(bloque_inicio*k, bloque_fin*k)]
        # Seleccionar los valores de la primera columna (primer sensor)
        valores_sensor_1 = df_filtrado.iloc[:, n_col]
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

def create_df_global(df):
    num_sensores = 14
    num_bloques = df['bloque'].max()
    # Creamos una lista para almacenar las filas del DataFrame global
    filas_df_global = []
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
    return df_global
    
    # Es importante mencionar que esta df_global tiene como elementos un conjunto de datos.
    # Y aunque pueden graficar las señales, se ha perdido la información sobre los "0", es decir el no agarre.
    # Por lo tanto, un elemento contiene 12 repeticiones de un tipo de agarre de un tipo de sensor.
    # Pero ya no se indica, dentro de esas repeticiones, cuándo hay agarre y cuándo no.
    # Este dataframe puede servir si solo interesa cada "serie de repeticiones" y no cuando hay agarre.
    # Pueden extraerse datos con, por ejemplo, df_global.iloc[0, 0] (fila 1, columna 1)
    # En este df, por ejemplo, pueden sacarse características por cada elemento, ya que son señales completas.

#---------------------------------------------------------------------------------

def create_df_pure(df):
    num_sensores = 14
    num_bloques = df['bloque'].max()
    # Creamos una lista para almacenar las filas de df_pure
    filas_df_pure = []
    # Iterar sobre los conjuntos de 24 bloques
    for i in range(0, num_bloques, 24):
        # Filtrar los 24 bloques actuales
        df_bloques = df[df['bloque'].between(i + 1, i + 24)]
        # Filtrar los 12 bloques donde el tipo de agarre es 0
        df_agarre_0 = df_bloques[df_bloques['stimulus'] == 0]
        # Filtrar los 12 bloques donde el tipo de agarre es diferente a 0 (agarre característico)
        df_agarre_n = df_bloques[df_bloques['stimulus'] != 0]
        # Obtener el tipo de agarre característico (debería ser un solo número)
        agarre_n = df_agarre_n['stimulus'].unique()[0]  # Se asume que solo hay un número diferente de 0
        # Crear una lista para la fila de agarre 0 (primera fila)
        fila_agarre_0 = []
        # Crear una lista para la fila de agarre n (segunda fila)
        fila_agarre_n = []
        # Iterar sobre cada sensor (las primeras 14 columnas)
        for sensor in range(num_sensores):
            # Combinar los valores de los 12 bloques del agarre 0 en un solo DataFrame
            df_sensor_0 = df_agarre_0.iloc[:, sensor].reset_index(drop=True)
            # Añadir este DataFrame a la fila de agarre 0
            fila_agarre_0.append(df_sensor_0)
            
            # Combinar los valores de los 12 bloques del agarre n en un solo DataFrame
            df_sensor_n = df_agarre_n.iloc[:, sensor].reset_index(drop=True)
            # Añadir este DataFrame a la fila de agarre n
            fila_agarre_n.append(df_sensor_n)
        # Añadir el valor del agarre a la columna 15
        fila_agarre_0.append(0)  # Agarre 0
        fila_agarre_n.append(agarre_n)  # Agarre n
        # Añadir las dos filas al DataFrame final
        filas_df_pure.append(fila_agarre_0)
        filas_df_pure.append(fila_agarre_n)
    
    # Crear el DataFrame final df_pure con 14 columnas de sensores y la columna del agarre
    columnas = [f'Sensor_{i+1}' for i in range(num_sensores)] + ['Stimulus']
    df_pure = pd.DataFrame(filas_df_pure, columns=columnas)
    return df_pure
    
    # El dataframe df_pure genera, de una muestra de doce repeticiones, dos conjuntos de datos.
    # El primer conjunto de datos es una fusión de todos los estímulos 0 de los primeros 24 bloques.
    # El segundo conjunto de datos es una fusión de los estímulos no 0 de los mismos 24 bloques.
    # Entonces tenemos una señal de solo estímulos 0 o solo estímulos de número característico.
    # Esto es lo contrario de df_global, porque exhibe en una señal solo un tipo de agarre.
    # No existen aquí los periodos de descanso. Una señal o no tiene agarre, o tiene solo un agarre.
    # Podría probarse sacar características de aquí, siendo estas señales de agarre puro.
    # A diferencia de df_global, que en un conjunto de datos combina el no agarre con un tipo de agarre.

#---------------------------------------------------------------------------------

#Para visualizar un elemento de df_global o df_pure
def simple_viewer(df_type,row=0,column=0):
    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    val = df_type.iloc[row, column] # [0,0] es fila 1 columna 1
    axs.plot(val.index, val)
    plt.show()

#---------------------------------------------------------------------------------

# Como info adicional, si se observa la señal del primer sensor para el primer tipo de agarre
# extrayendo los datos [0,0] de df_global (el código está comentado arriba) y luego se generan
# las gráficas de df_pure en la ubicación [0,0] y luego la gráfica en [1,0], se podrá ver
# que estas dos últimas juntas dan la primera, y es lo esperado pues una parte representa
# la posición sin agarre, y la otra del agarre característico, de modo que juntas forman la
# señal completa.

#---------------------------------------------------------------------------------

def create_df_block(df):
    num_sensores = 14
    num_bloques = df['bloque'].max()
    filas_df_block = []
    # Iteramos sobre cada bloque
    for i in range(1, num_bloques + 1):
        fila = []
        # Filtramos los datos del bloque actual
        df_filtrado = df[df['bloque'] == i]
        # Iteramos sobre cada sensor (las primeras 14 columnas)
        for sensor in range(num_sensores):
            # Creamos un DataFrame para los valores del sensor en este bloque
            df_sensor = df_filtrado.iloc[:, sensor]
            # Añadimos este DataFrame a la fila
            fila.append(df_sensor.values)
        # Obtener el valor del estímulo (penúltima columna)
        estimulo = df_filtrado['stimulus'].iloc[0]
        # Añadimos la fila con los valores de los sensores y el estímulo
        fila.append(estimulo)
        # Añadimos esta fila a la lista de filas globales
        filas_df_block.append(fila)
    # Crear el DataFrame global con las filas de sensores y la columna del estímulo
    columnas = [f'Sensor_{i+1}' for i in range(num_sensores)] + ['Stimulus']
    df_block = pd.DataFrame(filas_df_block, columns=columnas)
    
    return df_block
