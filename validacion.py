import os
import numpy as np
import wfdb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

def load_and_preprocess_single_ecg(file_path):
    """
    Carga y preprocesa una única señal de ECG desde un archivo .dat/.hea.
    """
    record_name = file_path.replace('.dat', '')
    try:
        signals, fields = wfdb.rdsamp(record_name)
        if signals.shape[1] == 12:
            signal_to_process = signals[:4000, :]

            original_shape = signal_to_process.shape
            signal_reshaped = signal_to_process.reshape(-1, original_shape[-1])

            scaler = StandardScaler()
            scaled_signal_reshaped = scaler.fit_transform(signal_reshaped)

            scaled_signal = scaled_signal_reshaped.reshape(original_shape)
            return np.expand_dims(scaled_signal, axis=0)
        else:
            print(f"Advertencia: {record_name} no tiene 12 derivaciones. Saltando.")
            return None
    except Exception as e:
        print(f"Error al procesar {record_name}: {e}")
        return None

def extract_label_from_hea(hea_path):
    """
    Extrae la etiqueta de Chagas desde el archivo .hea.
    """
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('# Chagas label:'):
                    label_text = line.strip().split(':')[-1].strip()
                    return 1 if label_text.lower() == 'true' else 0
    except Exception as e:
        print(f"Error leyendo etiqueta en {hea_path}: {e}")
    return None  # Si no se encuentra etiqueta

if __name__ == "__main__":
    model_path = 'chagas_detection_cnn_model3.h5'
    test_data_path = 'test'

    if not os.path.exists(model_path):
        print(f"Error: El modelo '{model_path}' no se encontró.")
    else:
        print("Cargando modelo...")
        model = tf.keras.models.load_model(model_path)
        print("Modelo cargado.")

        if not os.path.exists(test_data_path):
            print(f"Error: La carpeta '{test_data_path}' no existe.")
        else:
            y_true = []
            y_pred = []

            print(f"\nProcesando señales en '{test_data_path}'...")
            for file in os.listdir(test_data_path):
                if file.endswith('.dat'):
                    record_name = file.replace('.dat', '')
                    full_record_path = os.path.join(test_data_path, record_name)
                    hea_path = os.path.join(test_data_path, record_name + '.hea')

                    true_label = extract_label_from_hea(hea_path)
                    if true_label is None:
                        print(f"Etiqueta no encontrada en {hea_path}. Saltando.")
                        continue

                    signal = load_and_preprocess_single_ecg(full_record_path)
                    if signal is not None:
                        prediction = model.predict(signal)[0][0]
                        pred_label = 1 if prediction >= 0.5 else 0

                        y_true.append(true_label)
                        y_pred.append(pred_label)

                        print(f"\nArchivo: {record_name}")
                        print(f"Etiqueta real: {true_label} | Predicción: {prediction:.4f} → {pred_label}")
                    else:
                        print(f"No se pudo procesar {record_name}.")

            # Calcular matriz de confusión
            print("\n=== MATRIZ DE CONFUSIÓN ===")
            print(confusion_matrix(y_true, y_pred))
            print("\n=== REPORTE DE CLASIFICACIÓN ===")
            print(classification_report(y_true, y_pred, digits=4))
