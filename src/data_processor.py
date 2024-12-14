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

def lectura(mode=1,base_path='../datasets/'):

    if mode == 1:
        n0 = 1
        n1 = 10
        n2 = 5
        n3 = 2
    elif mode == 2:
        n0 = 10
        n1 = 11
        n2 = 5
        n3 = 2
    elif mode == 3:
        n0 = 1
        n1 = 4
        n2 = 6
        n3 = 3
    elif mode == 4:
        n0 = 10
        n1 = 11
        n2 = 4
        n3 = 3
    else:
        n0 = 1
        n1 = 3
        n2 = 4
        n3 = 3

    # Ruta base donde se encuentran los archivos
    dfs = []
    # Iteramos sobre las combinaciones de S, D y T
    for i in range(n0, n1):  # S del n0 al n1-1
        for j in range(1, n2):  # D del 1 al n2-1
            for k in range(1, n3):  # T del 1 al n3-1
                file_path = f'{base_path}S{i}_D{j}_T{k}.parquet'
                try:
                    # Intentamos leer el archivo
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    dfs.append(df)  # Si lo encontramos, lo añadimos a la lista
                except FileNotFoundError:
                    # Si no encontramos el archivo, continuamos con el siguiente
                    continue
    # Concatenamos todos los DataFrames en uno solo
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns=[df.columns[8], df.columns[9]])
    # Para saber cuántos datos de los estímulos hay.
    # Conteo_por_estado = df[df.columns[-1]].value_counts()
    
    '''
    # Se añade la columna bloque de modo que se cuente el cambio de estímulo.
    c_estimulo = df.columns[-1]
    df['bloque'] = (df[c_estimulo] != df[c_estimulo].shift()).cumsum()
    # Filtrar las filas donde los bloques no estén en el grupo 62
    inicio_grupo_62 = (64 - 1) * 24 + 1  # Primer bloque del grupo 63
    fin_grupo_62 = 64 * 24               # Último bloque del grupo 63
    df = df[~df['bloque'].between(inicio_grupo_62, fin_grupo_62)]
    # Se eliminan las columnas de bloques existentes
    df = df.drop(columns=['bloque'])
    '''

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
    columnas = [f'Sensor_{i+1}' for i in range(num_sensores)] + ['stimulus']
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
    columnas = [f'Sensor_{i+1}' for i in range(num_sensores)] + ['stimulus']
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
            # Convertimos los valores del sensor en una pd.Series (mantiene el índice)
            df_sensor = pd.Series(df_filtrado.iloc[:, sensor].values)
            # Añadimos este DataFrame a la fila
            fila.append(df_sensor)     
        # Obtener el valor del estímulo (penúltima columna)
        estimulo = df_filtrado['stimulus'].iloc[0]
        # Añadimos la fila con los valores de los sensores y el estímulo
        fila.append(estimulo)
        # Añadimos esta fila a la lista de filas globales
        filas_df_block.append(fila)
    # Crear el DataFrame global con las filas de DataFrames de sensores y la columna del estímulo
    columnas = [f'sensor_{i+1}' for i in range(num_sensores)] + ['stimulus']
    df_block = pd.DataFrame(filas_df_block, columns=columnas)
    
    return df_block

#---------------------------------------------------------------------------------

def filter(df,fs=2000,lowcut=20,highcut=500,notch_freq=50,kernel_size=9):
    '''
    fs: Frecuencia de muestreo en Hz
    lowcut: Para eliminar frecuencias inferiores al valor espeificado
    highcut: Para eliminar frecuencias superiores al valor espeificado
    notch_freq: Frecuencia del filtro notch (50 Hz debido a la frecuencia eléctrica)
    num_std: Número de desviaciones estándar para reemplazar los valores atípicos,
    es decir los límites de corte tanto superior como inferior para las señales EMG
    '''
    functions = [(ff.butter_bandpass, {'lowcut': lowcut, 'highcut': highcut, 'fs': fs}),
                (ff.apply_notch_filter, {'notch_freq': notch_freq, 'fs': fs}),
                (ff.apply_median_filter, {'kernel_size': kernel_size}),]
    stimulus = df.iloc[:, -1]  # Última columna (el tipo de agarre)
    df = df.iloc[:, :-1]  # Todas las columnas excepto la última
    dim = df.shape
    for funct, params in tqdm(functions, desc="Procesando"):
        for i in range(dim[0]):
            for j in range(dim[1]):
                df.iloc[i,j] = funct(df.iloc[i,j], **params)
    df['stimulus'] = stimulus
    return df

#---------------------------------------------------------------------------------

def pre_normalize(emg_features):
    """
    Normaliza las señales EMG contenidas en un DataFrame, separando características y estímulos.

    Args:
        df: DataFrame donde las columnas de características contienen señales EMG (pd.Series, listas o arrays)

    Returns:
        DataFrame con señales normalizadas y estímulos conservados.
    """

    # Inicializar un DataFrame vacío para almacenar las características normalizadas
    emg_features_normalized = pd.DataFrame(index=emg_features.index)

    # Normalizar cada columna de características
    scaler = StandardScaler()
    
    for col in emg_features.columns:
        normalized_signals = []
        for signal in emg_features[col]:
            # Convertir la señal a array si es pd.Series
            if isinstance(signal, pd.Series):
                signal = signal.to_numpy()
            elif isinstance(signal, (list, np.ndarray)):
                signal = np.array(signal)
            else:
                raise ValueError(f"Celda en la columna '{col}' no contiene una señal válida.")

            # Normalizar la señal individualmente
            signal = signal.reshape(-1, 1)  # Convertir a 2D para StandardScaler

            normalized_signal = scaler.fit_transform(signal).flatten()
            normalized_signals.append(pd.Series(normalized_signal))

        # Guardar las señales normalizadas en la columna correspondiente
        emg_features_normalized[col] = normalized_signals

    return emg_features_normalized

#--------------------------------------------------------------------------------------

def gen_carac(smeg_preprocess_data, feature_funcs):
    smeg_dim = smeg_preprocess_data.shape

    # Creamos un dataframe con columnas para cada característica y sensor
    sensor_count = smeg_dim[1]  # Cantidad de sensores (número de columnas del dataframe original)
    feature_names = list(feature_funcs.keys())  # Nombres de las características a extraer

    # Creamos las columnas de características para cada sensor
    columns = [f'{feature}_{sensor+1}' for sensor in range(sensor_count) for feature in feature_names]

    # Dataframe final para almacenar las características
    features_data = pd.DataFrame(columns=columns)

    # Recorremos el dataset y extraemos las características
    for i in tqdm(range(smeg_dim[0]), desc="Extrayendo Características"):  # Recorremos cada fila
        features_row = {}  # Fila donde se guardarán las características

        for j in range(smeg_dim[1]):  # Recorremos cada celda (sensor) de la fila
            series_data = smeg_preprocess_data.iloc[i, j].to_numpy()  # Convertimos la celda a numpy array

            # Extraemos las características para esta celda (sensor)
            for name, func in feature_funcs.items():
                feature_name = f'{name}_{j+1}'  # Ej: 'std_1', 'rms_1' para el sensor 1
                features_row[feature_name] = func(series_data)

        # Añadimos la fila de características al dataframe final
        features_data.loc[len(features_data)] = features_row

    # Llenamos los datos NaN con 0, esto debido a la conversión de 0 a NaN
    # en los dataframe
    features_data.fillna(0, inplace=True)

    return features_data

#-------------------------------------------------------------------------------------------

def get_carac(df):
    stimulus_data = df.iloc[:, -1]  # Última columna (el tipo de agarre)
    smeg_preprocess_data = df.iloc[:, :-1]  # Todas las columnas excepto la última

    time_features = {
      'rms': ff.rms,
      'iemg': ff.iemg,
      'mav': ff.mav,
      'wl': ff.wl,
      'log_detec': ff.log_detec,
      'ssi': ff.ssi,
    }

    time_df = gen_carac(smeg_preprocess_data, time_features)

    # Características que pueden extraerse
    spectral_features = {
      'fft': ff.fast_fourier_trans,
      'psd': ff.power_spect_dens,
      'mf': ff.mean_frequency,
      'mdf': ff.mdf,
      'zc': ff.zc,
      
      'ssc': ff.ssc,
      #'stft': ff.stft_features,
      #'cwt': ff.cwt_features,
      'df': ff.dominant_frequency,
      'sk': ff.spectral_kurtosis,
      'br': ff.band_ratio,
    }

    pre_spect_df = pre_normalize(smeg_preprocess_data)
    del smeg_preprocess_data

    spect_df = gen_carac(pre_spect_df, spectral_features)
    del pre_spect_df

    result = pd.concat([time_df, spect_df, stimulus_data], axis=1)

    return result.reset_index(drop=True)

#---------------------------------------------------------------------------------

def normalize(df, scaler=None):
    emg_features = df.iloc[:, :-1]
    emg_stimulus = df.iloc[:, -1]
    # Inicializamos el estandarizador para hacer Z-score
    if not scaler:
        scaler_std = StandardScaler()
    # Estandarizamos las columnas de características
    if not scaler:
        emg_features_normalized = scaler_std.fit_transform(emg_features)
    else:
        emg_features_normalized = scaler.transform(emg_features)

    # Convertimos las características normalizadas de nuevo a DataFrame, conservando las columnas originales
    emg_features_normalized = pd.DataFrame(emg_features_normalized, columns=emg_features.columns, index=emg_features.index)
    # Añadimos la columna de 'stimulus' nuevamente al DataFrame normalizado
    df_normalized = pd.concat([emg_features_normalized, emg_stimulus], axis=1)
    
    if not scaler:
        return scaler_std, df_normalized
    else:
        return df_normalized

#---------------------------------------------------------------------------------

def balance(dataframe):
    # Contar el número de filas por categoría en 'stimulus'
    counts = dataframe["stimulus"].value_counts()
    # Determinar el mínimo de filas entre las categorías
    min_count = counts.min()
    
    # Iterar sobre cada categoría que tiene más filas que min_count
    for value, count in counts.items():
        if count > min_count:
            # Calcular cuántas filas deben eliminarse
            excess_count = count - min_count
            # Obtener los índices de las filas que exceden el mínimo para esta categoría
            indices_to_drop = dataframe[dataframe["stimulus"] == value].sample(n=excess_count, random_state=42).index
            # Eliminar las filas en exceso
            dataframe.drop(indices_to_drop, inplace=True)
    
    # Resetear el índice del dataframe
    dataframe.reset_index(drop=True, inplace=True)