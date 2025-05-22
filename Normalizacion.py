import os
import wfdb
import numpy as np
from scipy.signal import resample

# Paths
pos_dir = 'dataset_submuestreado/positivos'  # carpeta con .dat y .hea de casos positivos
neg_dir = 'dataset_submuestreado/negativos'  # carpeta con subcarpetas de negativos
out_dir = 'data_normal'       # carpeta destino
os.makedirs(out_dir, exist_ok=True)

# Parameters
target_fs = 400  # Hz
duration_sec = 10  # segundos
target_length = target_fs * duration_sec

# Función para procesar un archivo

def process_record(record_path, out_path):
    # Leer registro con wfdb
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal  # shape (n_samples, n_leads)
    fs = record.fs
    # Resample
    n_leads = sig.shape[1]
    sig_res = np.zeros((target_length, n_leads))
    # Calcular número de muestras actuales
    curr_length = sig.shape[0]
    # Resample a target_fs*dur
    desired = int(fs / target_fs * target_length) if fs != target_fs else curr_length
    # Primera resample a frecuencia deseada relativa
    sig_rs = resample(sig, int(curr_length * target_fs / fs), axis=0)
    # Ahora ajustar longitud a target_length
    if sig_rs.shape[0] >= target_length:
        sig_res = sig_rs[:target_length, :]
    else:
        sig_res[:sig_rs.shape[0], :] = sig_rs
    # Guardar como .npy por canal o como uno solo
    np.save(out_path, sig_res)

# Recorrer positivos
for fname in os.listdir(pos_dir):
    if fname.endswith('.hea'):
        name = fname[:-4]
        record_path = os.path.join(pos_dir, name)
        out_path = os.path.join(out_dir, name + '.npy')
        process_record(record_path, out_path)

# Recorrer negativos
for root, dirs, files in os.walk(neg_dir):
    for fname in files:
        if fname.endswith('.hea'):
            name = os.path.splitext(fname)[0]
            record_path = os.path.join(root, name)
            out_name = root.replace(neg_dir + os.sep, '').replace(os.sep, '_') + '_' + name
            out_path = os.path.join(out_dir, out_name + '.npy')
            process_record(record_path, out_path)

print('Procesamiento completado.')
