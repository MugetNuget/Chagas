import os
import numpy as np
import wfdb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def load_ecg_data(dataset_path):
    """
    Carga los datos de ECG desde las carpetas 'positivos' y 'negativos'.

    Args:
        dataset_path (str): Ruta a la carpeta principal 'dataset_normalizado'.

    Returns:
        tuple: Tupla que contiene:
            - np.array: Datos de ECG.
            - np.array: Etiquetas (1 para positivo, 0 para negativo).
            - list: Nombres de los archivos.
    """
    all_signals = []
    all_labels = []
    all_record_names = []

    # Cargar datos de la carpeta 'positivos'
    positive_path = os.path.join(dataset_path, 'positivos')
    for record_file in os.listdir(positive_path):
        if record_file.endswith('.dat'):
            record_name = record_file.replace('.dat', '')
            record_path = os.path.join(positive_path, record_name)
            try:
                # wfdb.rdrecord devuelve un objeto Record que contiene la señal
                # y wfdb.rdsamp devuelve las señales y metadatos
                signals, fields = wfdb.rdsamp(record_path)
                # Asegurarse de que la señal tiene 12 derivaciones y una longitud consistente
                # Si la longitud es variable, se necesitará un padding o truncamiento
                # Para este ejemplo, asumimos una longitud fija o la ajustamos.
                # Aquí tomaremos los primeros 4000 puntos como en el ejemplo del .hea
                # Ajusta esto si tus señales tienen longitudes diferentes
                if signals.shape[1] == 12:
                    all_signals.append(signals[:4000, :]) # Tomar los primeros 4000 muestras
                    all_labels.append(1)  # Positivo para Chagas
                    all_record_names.append(record_name)
                else:
                    print(f"Advertencia: El archivo {record_name} no tiene 12 derivaciones. Saltando.")
            except Exception as e:
                print(f"Error al cargar {record_path}: {e}")

    # Cargar datos de la carpeta 'negativos'
    negative_path = os.path.join(dataset_path, 'negativos')
    for record_file in os.listdir(negative_path):
        if record_file.endswith('.dat'):
            record_name = record_file.replace('.dat', '')
            record_path = os.path.join(negative_path, record_name)
            try:
                signals, fields = wfdb.rdsamp(record_path)
                if signals.shape[1] == 12:
                    all_signals.append(signals[:4000, :]) # Tomar los primeros 4000 muestras
                    all_labels.append(0)  # Negativo para Chagas
                    all_record_names.append(record_name)
                else:
                    print(f"Advertencia: El archivo {record_name} no tiene 12 derivaciones. Saltando.")
            except Exception as e:
                print(f"Error al cargar {record_path}: {e}")

    return np.array(all_signals), np.array(all_labels), all_record_names

def preprocess_data(X):
    """
    Normaliza los datos de ECG utilizando StandardScaler.

    Args:
        X (np.array): Datos de ECG a normalizar.

    Returns:
        np.array: Datos de ECG normalizados.
    """
    # Reshape para aplicar StandardScaler a todas las derivaciones y muestras
    original_shape = X.shape
    X_reshaped = X.reshape(-1, original_shape[-1]) # (num_samples * timesteps, num_leads)

    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)

    # Volver a la forma original
    X_scaled = X_scaled_reshaped.reshape(original_shape)
    return X_scaled

def build_cnn_model(input_shape):

    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Salida binaria para clasificación (Chagas/No Chagas)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Ruta a tu carpeta principal 'dataset_normalizado'
    dataset_path = 'dataset_normalizado'
    model_save_path = 'chagas_detection_cnn_model3.h5'

    print("Cargando datos de ECG...")
    X, y, record_names = load_ecg_data(dataset_path)

    if len(X) == 0:
        print("No se encontraron datos de ECG. Asegúrate de que las carpetas 'positivos' y 'negativos' existan y contengan archivos .dat y .hea válidos.")
    else:
        print(f"Se cargaron {len(X)} muestras de ECG.")
        print("Preprocesando datos...")
        X_processed = preprocess_data(X)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Forma de los datos de entrenamiento: {X_train.shape}")
        print(f"Forma de las etiquetas de entrenamiento: {y_train.shape}")
        print(f"Forma de los datos de prueba: {X_test.shape}")
        print(f"Forma de las etiquetas de prueba: {y_test.shape}")

        # Obtener la forma de entrada para el modelo CNN
        input_shape = (X_train.shape[1], X_train.shape[2]) # (timesteps, num_leads)

        print("Construyendo el modelo CNN...")
        model = build_cnn_model(input_shape)
        model.summary()

        # Callbacks para guardar el mejor modelo y detener el entrenamiento temprano
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        print("Entrenando el modelo...")
        history = model.fit(X_train, y_train,
                            epochs=50, # Puedes ajustar el número de épocas
                            batch_size=32,
                            validation_split=0.1, # Usar una parte del conjunto de entrenamiento como validación
                            callbacks=[checkpoint, early_stopping])

        print(f"Modelo entrenado y guardado en: {model_save_path}")

        # Evaluar el modelo en el conjunto de prueba
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.4f}")
        print(f"Pérdida del modelo en el conjunto de prueba: {loss:.4f}")

        # Graficar el historial de entrenamiento
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
        plt.title('Precisión del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.title('Pérdida del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.tight_layout()
        plt.show()