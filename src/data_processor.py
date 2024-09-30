from tqdm import tqdm
import filters_and_features as ff

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
    # Code
    return True