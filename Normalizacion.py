import os
import wfdb
import numpy as np
from scipy.signal import resample
from shutil import copyfile

# Configuraciones
ORIG_PATH = 'dataset_submuestreado'
DEST_PATH = 'dataset_normalizado'
TARGET_FS = 400
DURATION_SEC = 10
TARGET_LENGTH = TARGET_FS * DURATION_SEC

# Crear directorios de destino
os.makedirs(os.path.join(DEST_PATH, 'positivos'), exist_ok=True)
os.makedirs(os.path.join(DEST_PATH, 'negativos'), exist_ok=True)

def normalize_signal(signal):
    # Normalización z-score por derivación
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True) + 1e-6
    return (signal - mean) / std

def process_record(filepath, label_folder):
    record_name = os.path.splitext(filepath)[0]
    full_path = os.path.join(ORIG_PATH, label_folder, record_name)

    # Leer señal y encabezado
    record = wfdb.rdrecord(full_path)
    signal = record.p_signal.T  # [12, N]
    fs = record.fs

    # Re-muestreo si es necesario
    if fs != TARGET_FS:
        num_samples = int(signal.shape[1] * TARGET_FS / fs)
        signal = resample(signal, num_samples, axis=1)

    # Recortar o rellenar
    if signal.shape[1] > TARGET_LENGTH:
        signal = signal[:, :TARGET_LENGTH]
    else:
        pad_width = TARGET_LENGTH - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), 'constant')

    # Normalizar
    signal = normalize_signal(signal)

    # Guardar .npy
    np.save(os.path.join(DEST_PATH, label_folder, record_name + '.npy'), signal)

    # Copiar el encabezado .hea (si lo necesitas para etiquetas)
    src_hea = os.path.join(ORIG_PATH, label_folder, record_name + '.hea')
    dst_hea = os.path.join(DEST_PATH, label_folder, record_name + '.hea')
    copyfile(src_hea, dst_hea)

# Procesar todos los archivos
for label_folder in ['positivos', 'negativos']:
    files = os.listdir(os.path.join(ORIG_PATH, label_folder))
    dat_files = sorted([f for f in files if f.endswith('.dat')])

    for dat_file in dat_files:
        try:
            print(f"Procesando {dat_file}...")
            process_record(dat_file, label_folder)
        except Exception as e:
            print(f"Error procesando {dat_file}: {e}")
