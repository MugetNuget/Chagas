import os
import wfdb
import matplotlib.pyplot as plt

# Configuración
BASE_ORIG_PATH = 'dataset_submuestreado'
BASE_NORM_PATH = 'dataset_normalizado'
LABELS = ['positivos', 'negativos']
CANAL = 'I'  # Cambiar si deseas otro canal (e.g., 'II', 'V1')
N_MUESTRAS = 4000  # muestras a graficar (10 segundos a 400 Hz)
N_ARCHIVOS = 3  # Número de archivos por clase a graficar

for label in LABELS:
    print(f"\nProcesando clase: {label}")

    ORIG_PATH = os.path.join(BASE_ORIG_PATH, label)
    NORM_PATH = os.path.join(BASE_NORM_PATH, label)

    # Obtener lista de archivos comunes
    archivos = sorted([f[:-4] for f in os.listdir(ORIG_PATH) if f.endswith('.dat')])
    archivos_norm = sorted([f[:-4] for f in os.listdir(NORM_PATH) if f.endswith('.dat')])
    archivos_comunes = list(set(archivos).intersection(archivos_norm))[:N_ARCHIVOS]

    for nombre in archivos_comunes:
        try:
            # Cargar registros
            record_orig = wfdb.rdrecord(os.path.join(ORIG_PATH, nombre))
            record_norm = wfdb.rdrecord(os.path.join(NORM_PATH, nombre))

            # Buscar índice del canal
            try:
                idx = record_orig.sig_name.index(CANAL)
            except ValueError:
                print(f"Canal {CANAL} no encontrado en {nombre}, se omite.")
                continue

            sig_orig = record_orig.p_signal[:, idx]
            sig_norm = record_norm.p_signal[:, idx]

            # Graficar
            plt.figure(figsize=(12, 4))
            plt.plot(sig_orig[:N_MUESTRAS], label='Original', alpha=0.8)
            plt.plot(sig_norm[:N_MUESTRAS], label='Normalizada', alpha=0.8)
            plt.title(f'[{label.upper()}] Comparación canal {CANAL} - {nombre}')
            plt.xlabel('Muestras (a 400 Hz)')
            plt.ylabel('Amplitud')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error procesando {nombre}: {e}")
