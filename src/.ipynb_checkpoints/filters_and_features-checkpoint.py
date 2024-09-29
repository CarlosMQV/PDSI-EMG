import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy import signal
from scipy.fftpack import fft

# Filtro Butterworth pasa-altos y pasa-bajos
def butter_bandpass(data, lowcut, highcut, fs, order=4):
  """
  Aplica un filtro Butterworth pasa-altos y pasa-bajos a los datos.

  Parámetros:
  data (pd.Series): Dataset con las señales a filtrar.
  lowcut (float): Frecuencia de corte inferior para eliminar frecuencias bajas.
  highcut (float): Frecuencia de corte superior para eliminar frecuencias altas.
  fs (int): Frecuencia de muestreo en Hz.
  order (int): Orden del filtro, por defecto es 4.

  Devuelve:
  pd.Series: La serie filtrada con el mismo índice y nombre que los datos de entrada.
  """
  nyq = 0.5 * fs  # Frecuencia de Nyquist
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  result = pd.Series(filtfilt(b, a, data), index=data.index, name=data.name)
  return result

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
  """
  Aplica un filtro notch a los datos para eliminar frecuencias específicas.

  Parámetros:
  data (pd.Series): Datos a filtrar.
  notch_freq (float): Frecuencia a eliminar (notch).
  fs (int): Frecuencia de muestreo en Hz.
  quality_factor (float): Factor de calidad del filtro, por defecto es 30.

  Devuelve:
  pd.Series: La serie filtrada con el mismo índice y nombre que los datos de entrada.
  """
  nyq = 0.5 * fs
  notch = notch_freq / nyq
  b, a = iirnotch(notch, quality_factor)
  result = pd.Series(filtfilt(b, a, data), index=data.index, name=data.name)
  return result

def replace_outliers_with_threshold(series_data, num_std):
  """
  Reemplaza los valores atípicos en una serie de datos con límites de umbral definidos por la media y desviación estándar.

  Parámetros:
  series_data (pd.Series): La serie de datos en la que se buscarán valores atípicos.
  num_std (float): Número de desviaciones estándar para definir los límites superior e inferior.

  Devuelve:
  pd.Series: La serie con los valores atípicos reemplazados por los límites del umbral.
  """
  # Calculamos la media y la desviación estándar de la serie
  mean_signal = series_data.mean()
  std_signal = series_data.std()

  # Definimos los umbrales superior e inferior
  upper_threshold = mean_signal + num_std * std_signal
  lower_threshold = mean_signal - num_std * std_signal

  # Reemplazamos los valores fuera de los umbrales con el límite del umbral
  clipped_data = np.clip(series_data, lower_threshold, upper_threshold)

  # Convertimos los resultados a una Serie de pandas, manteniendo el nombre de la columna original
  result = pd.Series(clipped_data, index=series_data.index, name=series_data.name)
  return result

def std(data):
  """
  Calcula la desviación estándar de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la desviación estándar.

  Devuelve:
  float: La desviación estándar de los datos.
  """
  return np.std(data)

def rms(data):
  """
  Calcula la raíz cuadrática media (RMS) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular el RMS.

  Devuelve:
  float: La raíz cuadrática media de los datos.
  """
  return np.sqrt(np.mean(np.square(data)))

def iemg(data):
  """
  Calcula la integral de la señal EMG (iEMG).

  Parámetros:
  data (np.ndarray): Datos de la señal EMG.

  Devuelve:
  float: La suma de los valores absolutos de la señal EMG.
  """
  return np.sum(np.abs(data))

def mav(data):
  """
  Calcula el valor absoluto medio (MAV) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular el MAV.

  Devuelve:
  float: El valor absoluto medio de los datos.
  """
  return np.mean(np.abs(data))

def wl(data):
  """
  Calcula la longitud de la forma de onda (WL) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la longitud de la forma de onda.

  Devuelve:
  float: La longitud de la forma de onda.
  """
  return np.sum(np.abs(np.diff(data)))

def log_detec(data):
  """
  Calcula el detector logarítmico de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular el detector logarítmico.

  Devuelve:
  float: El valor del detector logarítmico.
  """
  return np.exp(np.mean(np.log(np.abs(data) + 1e-10)))

def ssi(data):
  """
  Calcula la integral de la señal cuadrática (SSI) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la integral de la señal cuadrática.

  Devuelve:
  float: La suma de los valores cuadrados de los datos.
  """
  return np.sum(np.square(data))

def fast_fourier_trans(data):
  """
  Calcula la transformada rápida de Fourier (FFT) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la FFT.

  Devuelve:
  float: El promedio de los valores absolutos de la FFT.
  """
  return np.mean(np.abs(fft(data)))

def power_spect_dens(data):
  """
  Calcula la densidad de potencia espectral (PSD) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la densidad de potencia espectral.
  
  Devuelve:
  float: El promedio de la densidad de potencia espectral.
  """
  f, psd = signal.welch(data, fs=2000, nperseg=1024)
  return np.mean(psd)

def mdf(data):
  """
  Calcula la frecuencia de potencia mediana (MDF) de los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular la frecuencia de potencia mediana.

  Devuelve:
  float: El índice de la frecuencia de potencia mediana.
  """
  f, psd = signal.welch(data, fs=2000, nperseg=1024)
  cumulative_power = np.cumsum(psd)
  total_power = cumulative_power[-1]
  mdf_index = np.where(cumulative_power >= total_power / 2)[0][0]
  return mdf_index

def zc(data):
  """
  Calcula el número de cruces por cero (ZC) en los datos.

  Parámetros:
  data (np.ndarray): Datos para calcular los cruces por cero.

  Devuelve:
  int: El número de cruces por cero en los datos.
  """
  return len(np.where(np.diff(np.sign(data)))[0])