import os
import wfdb
import numpy as np
from scipy.signal import resample

fs_target = 400
duration_target = 10
samples_target = fs_target * duration_target

input_root = 'dataset_submuestreado'
output_root = 'dataset_resampled'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_hea_metadata(hea_path):
    """Leer archivo .hea y extraer etiquetas # Age, # Sex, # Chagas label y demás líneas"""
    with open(hea_path, 'r') as f:
        lines = f.readlines()

    metadata_lines = []
    for line in lines:
        if line.startswith('# Age') or line.startswith('# Sex') or line.startswith('# Chagas label'):
            metadata_lines.append(line.strip())
    return metadata_lines, lines

def write_hea_with_metadata(output_hea_path, wfdb_header_lines, metadata_lines):
    """
    wfdb_header_lines: líneas que genera wfdb (para la cabecera estándar)
    metadata_lines: líneas de etiquetas originales que queremos conservar
    """
    # Escribimos el archivo .hea completo:
    # Primero las líneas estándar de wfdb
    with open(output_hea_path, 'w') as f:
        for line in wfdb_header_lines:
            f.write(line + '\n')
        # Luego añadimos las etiquetas originales
        for meta_line in metadata_lines:
            f.write(meta_line + '\n')

def process_record(record_path, output_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    fs_orig = record.fs
    
    samples_orig_target = int(duration_target * fs_orig)
    if signal.shape[0] > samples_orig_target:
        signal_cropped = signal[:samples_orig_target, :]
    else:
        pad_length = samples_orig_target - signal.shape[0]
        signal_cropped = np.pad(signal, ((0, pad_length), (0,0)), 'constant')

    signal_resampled = resample(signal_cropped, samples_target, axis=0)

    # Guardar con wfdb
    wfdb.wrsamp(
        record_name=output_path,
        fs=fs_target,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=signal_resampled,
        fmt=record.fmt
    )

    # Ahora modificamos el .hea para agregar las etiquetas originales
    input_hea_path = record_path + '.hea'
    output_hea_path = output_path + '.hea'

    metadata_lines, original_hea_lines = read_hea_metadata(input_hea_path)

    # Las líneas que creó wfdb en el archivo .hea recien creado
    with open(output_hea_path, 'r') as f:
        wfdb_header_lines = [line.strip() for line in f.readlines() if not line.startswith('#')]

    # Sobreescribir el .hea con las líneas estándar + las etiquetas
    write_hea_with_metadata(output_hea_path, wfdb_header_lines, metadata_lines)

    print(f'Procesado {record_path} -> {output_path}')

def main():
    for label in ['positivos', 'negativos']:
        input_folder = os.path.join(input_root, label)
        output_folder = os.path.join(output_root, label)
        ensure_dir(output_folder)

        for file in os.listdir(input_folder):
            if file.endswith('.hea'):
                record_name = file[:-4]
                input_record_path = os.path.join(input_folder, record_name)
                output_record_path = os.path.join(output_folder, record_name)
                try:
                    process_record(input_record_path, output_record_path)
                except Exception as e:
                    print(f'Error procesando {input_record_path}: {e}')

if __name__ == '__main__':
    main()
