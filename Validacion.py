import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

DATASET_PATH = 'dataset_normalizado'
DERIVACIONES = 12
TARGET_LENGTH = 4000  # 10 segundos * 400 Hz

def check_shapes():
    print("Verificando formas de los arrays...")
    errores = 0
    for folder in ['positivos', 'negativos']:
        path = os.path.join(DATASET_PATH, folder)
        for file in os.listdir(path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(path, file))
                if data.shape != (DERIVACIONES, TARGET_LENGTH):
                    print(f"❌ {file} en {folder} tiene forma incorrecta: {data.shape}")
                    errores += 1
    if errores == 0:
        print("✅ Todas las señales tienen la forma correcta.")
    else:
        print(f"⚠️ Se encontraron {errores} señales con forma incorrecta.")

def plot_random_signal():
    print("\nVisualizando una señal aleatoria...")
    folder = random.choice(['positivos', 'negativos'])
    path = os.path.join(DATASET_PATH, folder)
    archivo = random.choice([f for f in os.listdir(path) if f.endswith('.npy')])
    signal = np.load(os.path.join(path, archivo))
    
    plt.figure(figsize=(12, 10))
    for i in range(DERIVACIONES):
        plt.subplot(6, 2, i + 1)
        plt.plot(signal[i])
        plt.title(f'Derivación {i+1}')
        plt.tight_layout()
    plt.suptitle(f'Señal: {archivo} ({folder})', y=1.02)
    plt.show()

def extract_labels():
    print("\nExtrayendo etiquetas desde archivos .hea...")
    metadata = []
    for label_folder in ['positivos', 'negativos']:
        folder_path = os.path.join(DATASET_PATH, label_folder)
        for file in os.listdir(folder_path):
            if file.endswith('.hea'):
                path = os.path.join(folder_path, file)
                with open(path, 'r') as f:
                    lines = f.readlines()
                info = {'file': file.replace('.hea', ''), 'label': label_folder}
                for line in lines:
                    lower_line = line.lower()
                    if 'chagas' in lower_line:
                        info['chagas'] = line.strip().split()[-1]
                    if 'age' in lower_line:
                        info['age'] = line.strip().split()[-1]
                    if 'sex' in lower_line:
                        info['sex'] = line.strip().split()[-1]
                metadata.append(info)
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(DATASET_PATH, 'metadatos_chagas.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Metadatos guardados en {csv_path}")
    print(df.head())

def check_normalization():
    print("\nVerificando normalización z-score...")
    means = []
    stds = []
    for folder in ['positivos', 'negativos']:
        path = os.path.join(DATASET_PATH, folder)
        for file in os.listdir(path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(path, file))
                means.append(np.mean(data))
                stds.append(np.std(data))
    print(f"Media promedio: {np.mean(means):.6f}")
    print(f"Desviación estándar promedio: {np.mean(stds):.6f}")

def check_balance():
    positivos = len([f for f in os.listdir(os.path.join(DATASET_PATH, 'positivos')) if f.endswith('.npy')])
    negativos = len([f for f in os.listdir(os.path.join(DATASET_PATH, 'negativos')) if f.endswith('.npy')])
    print(f"\nBalance de clases:\nPositivos: {positivos}\nNegativos: {negativos}")

if __name__ == "__main__":
    check_shapes()
    plot_random_signal()
    extract_labels()
    check_normalization()
    check_balance()
