import os
import wfdb
import numpy as np
from scipy.signal import resample

# Rutas
dir_positivos = 'samitrop_output'    # carpeta con .dat y .hea de casos positivos
dir_negativos = 'samitrop_output'    # carpeta con subcarpetas de casos negativos
dir_salida = 'data_normal'           # carpeta destino
os.makedirs(dir_salida, exist_ok=True)

# Parámetros
frecuencia_objetivo = 400            # Hz
duracion_segundos = 10               # segundos
longitud_objetivo = frecuencia_objetivo * duracion_segundos

def procesar_registro(ruta_registro, ruta_salida):
    """Lee un registro WFDB, remuestrea a la frecuencia y duración deseadas, y guarda como .npy."""
    # Cargar registro con wfdb
    registro = wfdb.rdrecord(ruta_registro)
    senal = registro.p_signal             # array (muestras, derivaciones)
    frecuencia_muestreo = registro.fs
    num_derivaciones = senal.shape[1]
    # Array para señal remuestreada con longitud fija
    senal_resampleada = np.zeros((longitud_objetivo, num_derivaciones))
    # Longitud actual de la señal
    longitud_actual = senal.shape[0]
    # Primer remuestreo relativo a frecuencia_objetivo
    senal_rs = resample(senal,
                        int(longitud_actual * frecuencia_objetivo / frecuencia_muestreo),
                        axis=0)
    # Ajustar la longitud al valor objetivo
    if senal_rs.shape[0] >= longitud_objetivo:
        senal_resampleada = senal_rs[:longitud_objetivo, :]
    else:
        senal_resampleada[:senal_rs.shape[0], :] = senal_rs
    # Guardar array .npy
    np.save(ruta_salida, senal_resampleada)

# Procesar todos los registros positivos
for archivo in os.listdir(dir_positivos):
    if archivo.endswith('.hea'):
        nombre = archivo[:-4]
        ruta_registro = os.path.join(dir_positivos, nombre)
        ruta_salida = os.path.join(dir_salida, nombre + '.npy')
        procesar_registro(ruta_registro, ruta_salida)

# Procesar todos los registros negativos (buscando recursivamente)
for carpeta, subcarpetas, archivos in os.walk(dir_negativos):
    for archivo in archivos:
        if archivo.endswith('.hea'):
            nombre = os.path.splitext(archivo)[0]
            ruta_registro = os.path.join(carpeta, nombre)
            # Crear nombre de salida que incluya subcarpeta de origen
            prefijo = carpeta.replace(dir_negativos + os.sep, '').replace(os.sep, '_')
            nombre_salida = f"{prefijo}_{nombre}"
            ruta_salida = os.path.join(dir_salida, nombre_salida + '.npy')
            procesar_registro(ruta_registro, ruta_salida)

print('Procesamiento completado.')
