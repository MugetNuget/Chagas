import os
import random
import numpy as np
import wfdb
from scipy.signal import resample

# --- PARÁMETROS DE USUARIO ---
DIR_ENTRADA = 'samitrop_output'        # carpeta con tus 815 pares .hea/.dat originales
DIR_SALIDA = 'samitrop_aumentado'      # carpeta que contendrá los 3000 registros aumentados
OBJETIVO = 3000                        # total de registros deseados

# --- CONFIGURACIÓN ---
os.makedirs(DIR_SALIDA, exist_ok=True)

# 1. obtén todos los nombres de registros originales (sin extensión)
registros = [fn[:-4] for fn in os.listdir(DIR_ENTRADA) if fn.lower().endswith('.hea')]
if not registros:
    raise RuntimeError(f"No se encontraron archivos .hea en {DIR_ENTRADA!r}")

def leer_comentarios_encabezado(ruta_hea):
    """Lee y limpia las líneas de metadatos de un archivo .hea (eliminando '# ')."""
    comentarios = []
    with open(ruta_hea, 'r') as f:
        for linea in f:
            if linea.startswith('#'):
                comentarios.append(linea.lstrip('# ').rstrip('\n'))
    return comentarios

# 2. define transformaciones de aumento
def agregar_ruido_gaussiano(senal, nivel_ruido=0.02):
    return senal + np.random.normal(0, nivel_ruido, senal.shape)

def escalar_amplitud(senal, minimo=0.95, maximo=1.05):
    return senal * np.random.uniform(minimo, maximo)

def desplazamiento_temporal(senal, max_frac=0.05):
    n = senal.shape[0]
    desplaz = int(np.random.uniform(-max_frac, max_frac) * n)
    return np.roll(senal, desplaz, axis=0)

def estiramiento_temporal(senal, minimo=0.95, maximo=1.05):
    factor = np.random.uniform(minimo, maximo)
    estirada = resample(senal, int(senal.shape[0] * factor), axis=0)
    # recorta o rellena para mantener longitud original
    if estirada.shape[0] > senal.shape[0]:
        return estirada[:senal.shape[0]]
    relleno = np.zeros((senal.shape[0] - estirada.shape[0], senal.shape[1]))
    return np.vstack([estirada, relleno])

def deriva_linea_base(senal, max_deriva=0.02):
    n, _ = senal.shape
    t = np.linspace(0, 2*np.pi, n)
    deriva = (np.sin(t * np.random.uniform(0.5, 2.0)) *
              np.random.uniform(-max_deriva, max_deriva))[:, None]
    return senal + deriva

def aumentar(senal):
    """Aplica aleatoriamente 2–3 transformaciones de aumento."""
    funciones = [
        agregar_ruido_gaussiano,
        escalar_amplitud,
        desplazamiento_temporal,
        estiramiento_temporal,
        deriva_linea_base
    ]
    for func in random.sample(funciones, k=random.randint(2, 3)):
        senal = func(senal)
    return senal

# 3. copia los originales a DIR_SALIDA (para tener tus 815 bases + aumentos)
for reg in registros:
    for ext in ('.hea', '.dat'):
        origen = os.path.join(DIR_ENTRADA, reg + ext)
        destino = os.path.join(DIR_SALIDA, reg + ext)
        if not os.path.exists(destino):
            with open(origen, 'rb') as f_or, open(destino, 'wb') as f_dst:
                f_dst.write(f_or.read())

# 4. genera registros aumentados hasta alcanzar OBJETIVO
contador = len(registros)
while contador < OBJETIVO:
    orig = random.choice(registros)
    registro = wfdb.rdrecord(os.path.join(DIR_ENTRADA, orig))
    senal = registro.p_signal.copy()
    senal_aumentada = aumentar(senal)

    # conserva las líneas de metadatos
    ruta_hea = os.path.join(DIR_ENTRADA, orig + '.hea')
    comentarios = getattr(registro, 'comments', None) or leer_comentarios_encabezado(ruta_hea)

    nuevo_nombre = f"{orig}_aug{contador}"
    wfdb.wrsamp(
        record_name=nuevo_nombre,     # solo el nombre base en la cabecera
        fs=registro.fs,
        units=registro.units,
        sig_name=registro.sig_name,
        p_signal=senal_aumentada,
        fmt=registro.fmt,
        adc_gain=getattr(registro, 'adc_gain', None),
        baseline=getattr(registro, 'baseline', None),
        comments=comentarios,
        write_dir=DIR_SALIDA,         # aquí se escriben .hea/.dat
    )
    contador += 1

print(f"Hecho — {contador} registros (incluyendo {contador - len(registros)} aumentos) en '{DIR_SALIDA}'.")
