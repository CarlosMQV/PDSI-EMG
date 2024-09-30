from tqdm import tqdm
import filters_and_features as ff
import pandas as pd
import numpy as np

def filtrar(df,fs=2000,lowcut=15,highcut=500,notch_freq=50,num_std=6):
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
                (ff.replace_outliers_with_threshold, {'num_std': num_std}),]
    Stimulus = df.iloc[:, -1]  # Última columna (el tipo de agarre)
    df = df.iloc[:, :-1]  # Todas las columnas excepto la última
    dim = df.shape
    for funct, params in tqdm(functions, desc="Procesando"):
        for i in range(dim[0]):
            for j in range(dim[1]):
                df.iloc[i,j] = funct(df.iloc[i,j], **params)
    df['Stimulus'] = Stimulus
    return df

def gen_carac(df):
    stimulus_data = df.iloc[:, -1]  # Última columna (el tipo de agarre)
    smeg_preprocess_data = df.iloc[:, :-1]  # Todas las columnas excepto la última

    smeg_dim = smeg_preprocess_data.shape

    # Características que pueden extraerse
    feature_funcs = {
        'rms': ff.rms,
        'iemg': ff.iemg,
        'mav': ff.mav,
        'wl': ff.wl,
        'log_detec': ff.log_detec,
        'ssi': ff.ssi,
        'fft': ff.fast_fourier_trans,
        'psd': ff.power_spect_dens,
        'mf': ff.mean_frequency,
        'mdf': ff.mdf,
        'zc': ff.zc,
    }
    # Creamos un dataframe con columnas para cada característica y sensor
    sensor_count = smeg_dim[1]  # Cantidad de sensores (número de columnas del dataframe original)
    feature_names = list(feature_funcs.keys())  # Nombres de las características a extraer

    # Creamos las columnas de características para cada sensor
    columns = [f'{feature}_{sensor+1}' for sensor in range(sensor_count) for feature in feature_names]
    columns.append('stimulus')  # Añadimos la columna para el estímulo

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

        # Añadimos el valor del estímulo correspondiente a esta fila
        features_row['stimulus'] = stimulus_data[i]

        # Añadimos la fila de características al dataframe final
        features_data.loc[len(features_data)] = features_row

    # Llenamos los datos NaN con 0, esto debido a la conversión de 0 a NaN
    # en los dataframe
    features_data.fillna(0, inplace=True)

    return features_data