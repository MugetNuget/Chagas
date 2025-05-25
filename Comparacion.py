import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import random

def load_and_preprocess(path, filename, target_fs=400, target_duration=10):
    record = wfdb.rdrecord(os.path.join(path, filename))
    signal = record.p_signal.T  # (12, muestras)
    original_fs = record.fs
    
    if original_fs != target_fs:
        n_samples = int(signal.shape[1] * target_fs / original_fs)
        signal = resample(signal, n_samples, axis=1)
    
    target_samples = target_fs * target_duration
    if signal.shape[1] > target_samples:
        signal = signal[:, :target_samples]
    elif signal.shape[1] < target_samples:
        pad_width = target_samples - signal.shape[1]
        signal = np.pad(signal, ((0,0),(0,pad_width)), 'constant')
    
    # Normalización z-score por canal
    signal_norm = (signal - np.mean(signal, axis=1, keepdims=True)) / np.std(signal, axis=1, keepdims=True)
    
    return signal, signal_norm

def plot_comparison(original, normalized, title):
    plt.figure(figsize=(15, 20))
    for i in range(12):
        plt.subplot(12, 2, 2*i+1)
        plt.plot(original[i])
        plt.title(f'Derivación {i+1} - Original')
        plt.grid(True)
        
        plt.subplot(12, 2, 2*i+2)
        plt.plot(normalized[i])
        plt.title(f'Derivación {i+1} - Normalizada')
        plt.grid(True)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    base_path_original = 'dataset_submuestreado'
    base_path_normalized = 'dataset_normalizado'
    
    # Elegir aleatoriamente carpeta
    folder = random.choice(['positivos', 'negativos'])
    path_orig = os.path.join(base_path_original, folder)
    path_norm = os.path.join(base_path_normalized, folder)
    
    # Listar señales (sin extensión)
    archivos = [f[:-4] for f in os.listdir(path_orig) if f.endswith('.dat')]
    archivo_elegido = random.choice(archivos)
    
    print(f"Comparando señal: {archivo_elegido} de la carpeta '{folder}'")
    
    # Cargar y procesar señal original
    signal_orig, signal_proc = load_and_preprocess(path_orig, archivo_elegido)
    
    # Cargar señal normalizada guardada
    signal_norm = np.load(os.path.join(path_norm, archivo_elegido + '.npy'))
    
    # Graficar comparación
    plot_comparison(signal_orig, signal_norm, f'Señal: {archivo_elegido} ({folder})')
