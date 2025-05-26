import os
import random
import shutil

bases_dir=['samitrop_aumentado','ptbxl_output']

for base in bases_dir:
    root_dir = base
    extensions = {'.hea', '.dat'}
    count = 0


    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in extensions:
                count += 1

    print('\n')
    print('################################################################')
    print(f"Archivos .hea/.dat encontrados en {base} es: {count}")
    print(f"Señales (pares) estimadas: {count // 2}")
    print('################################################################')
    print('\n')

positivos_dir= bases_dir[0]

# Carpeta destino balanceada
balanceado_dir = 'dataset_submuestreado'
positivos_dest = os.path.join(balanceado_dir, 'positivos')
negativos_dest = os.path.join(balanceado_dir, 'negativos')

# Crear carpetas destino si no existen
os.makedirs(positivos_dest, exist_ok=True)
os.makedirs(negativos_dest, exist_ok=True)

positivos_hea = [f for f in os.listdir(positivos_dir) if f.endswith('.hea')]

positivos_files = []
for hea in positivos_hea:
    base = hea[:-4]
    dat = base + '.dat'
    if os.path.exists(os.path.join(positivos_dir, dat)):
        positivos_files.append(base)
print(f'Total positivos encontrados: {len(positivos_files)}')

negativos_dir= bases_dir[1]
negativos_files = []
for root, dirs, files in os.walk(negativos_dir):
    hea_files = [f for f in files if f.endswith('.hea')]
    for hea in hea_files:
        base = hea[:-4]
        dat = base + '.dat'
        if dat in files:
            # Guarda ruta completa y base para copiar después
            negativos_files.append((root, base))

print(f'Total negativos encontrados: {len(negativos_files)}')

#Submuestrear negativos
n_neg_sample = 3000
neg_sampled = random.sample(negativos_files, min(n_neg_sample, len(negativos_files)))

#Copiar positivos a destino
for base in positivos_files:
    shutil.copy(os.path.join(positivos_dir, base+'.hea'), positivos_dest)
    shutil.copy(os.path.join(positivos_dir, base+'.dat'), positivos_dest)

#Copiar negativos submuestreados a destino
for root, base in neg_sampled:
    shutil.copy(os.path.join(root, base+'.hea'), negativos_dest)
    shutil.copy(os.path.join(root, base+'.dat'), negativos_dest)

print(f'Archivos balanceados copiados: Positivos={len(positivos_files)}, Negativos={len(neg_sampled)}')
