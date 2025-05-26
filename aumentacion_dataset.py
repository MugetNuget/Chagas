import os
import random
import numpy as np
import wfdb
from scipy.signal import resample

# --- USER PARAMETERS ---
INPUT_DIR = 'samitrop_output'       # folder with your 815 .hea/.dat pairs
OUTPUT_DIR = 'samitrop_aumentado'   # will hold 3000 augmented records
TARGET = 3000                       # total records desired

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. find all original record names (basenames without extension)
records = [fn[:-4] for fn in os.listdir(INPUT_DIR) if fn.lower().endswith('.hea')]
if not records:
    raise RuntimeError(f"No .hea files found in {INPUT_DIR!r}")

def read_header_comments(hea_path):
    """Read and strip the leading '# ' from metadata lines in a .hea file."""
    comments = []
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                comments.append(line.lstrip('# ').rstrip('\n'))
    return comments

# 2. define augmentation transforms
def add_gaussian_noise(sig, noise_level=0.02):
    return sig + np.random.normal(0, noise_level, sig.shape)

def scale_amplitude(sig, lo=0.95, hi=1.05):
    return sig * np.random.uniform(lo, hi)

def time_shift(sig, max_frac=0.05):
    n = sig.shape[0]
    shift = int(np.random.uniform(-max_frac, max_frac) * n)
    return np.roll(sig, shift, axis=0)

def time_stretch(sig, lo=0.95, hi=1.05):
    stretch = np.random.uniform(lo, hi)
    stretched = resample(sig, int(sig.shape[0] * stretch), axis=0)
    # crop or pad to original length
    if stretched.shape[0] > sig.shape[0]:
        return stretched[:sig.shape[0]]
    pad = np.zeros((sig.shape[0] - stretched.shape[0], sig.shape[1]))
    return np.vstack([stretched, pad])

def baseline_wander(sig, max_wander=0.02):
    n, _ = sig.shape
    t = np.linspace(0, 2*np.pi, n)
    drift = (np.sin(t * np.random.uniform(0.5, 2.0)) *
             np.random.uniform(-max_wander, max_wander))[:, None]
    return sig + drift

def augment(sig):
    """Randomly apply 2–3 of the augmentations."""
    funcs = [add_gaussian_noise, scale_amplitude, time_shift, time_stretch, baseline_wander]
    for f in random.sample(funcs, k=random.randint(2,3)):
        sig = f(sig)
    return sig

# 3. copy originals into OUTPUT_DIR (so you end up with your 815 bases + augments)
for rec in records:
    for ext in ('.hea', '.dat'):
        src = os.path.join(INPUT_DIR, rec + ext)
        dst = os.path.join(OUTPUT_DIR, rec + ext)
        if not os.path.exists(dst):
            with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                f_dst.write(f_src.read())

# 4. now generate augmentations up to TARGET
count = len(records)
while count < TARGET:
    orig = random.choice(records)
    rec = wfdb.rdrecord(os.path.join(INPUT_DIR, orig))
    sig = rec.p_signal.copy()
    aug_sig = augment(sig)

    # preserve the metadata lines
    hea_path = os.path.join(INPUT_DIR, orig + '.hea')
    comments = getattr(rec, 'comments', None) or read_header_comments(hea_path)

    new_name = f"{orig}_aug{count}"
    wfdb.wrsamp(
        # only the basename goes into the header's first line
        record_name=new_name,
        fs=rec.fs,
        units=rec.units,
        sig_name=rec.sig_name,
        p_signal=aug_sig,
        fmt=rec.fmt,
        adc_gain=getattr(rec, 'adc_gain', None),
        baseline=getattr(rec, 'baseline', None),
        comments=comments,
        write_dir=OUTPUT_DIR,   # this is where .hea/.dat actually get written
    )
    count += 1

print(f"Done — {count} records (including {count - len(records)} augmentations) in '{OUTPUT_DIR}'.")
