import os
import wfdb
import numpy as np
from scipy.signal import resample

# ---- USER PARAMETERS ----
INPUT_DIR      = "dataset_submuestreado"
OUTPUT_DIR     = "dataset_standardized"
TARGET_FS      = 400               # Hz
DURATION_SEC   = 10                # seconds
TARGET_SAMPLES = TARGET_FS * DURATION_SEC  # 4000 samples

# make output folders
for label in ("positivos","negativos"):
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# ---- FUNCTIONS ----
def standardize_ecg(sig: np.ndarray, orig_fs: float) -> np.ndarray:
    """
    sig: shape (n_samples, n_leads)
    returns: shape (TARGET_SAMPLES, n_leads) resampled, clipped/padded, normalized
    """
    # 1) Resample
    if orig_fs != TARGET_FS:
        new_len = int(sig.shape[0] * TARGET_FS / orig_fs)
        sig = resample(sig, new_len, axis=0)

    # 2) Clip or pad
    if sig.shape[0] > TARGET_SAMPLES:
        sig = sig[:TARGET_SAMPLES]
    elif sig.shape[0] < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - sig.shape[0]
        sig = np.vstack([sig, np.zeros((pad, sig.shape[1]))])

    # 3) Magnitude normalization to [-1,1]
    max_vals = np.max(np.abs(sig), axis=0, keepdims=True) + 1e-6
    return sig / max_vals

def read_header_comments(hea_path: str) -> list[str]:
    """
    Reads all lines beginning with '#' from a .hea file,
    strips the leading '# ' and newline, and returns them as a list.
    """
    comments = []
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                comments.append(line.lstrip('# ').rstrip('\n'))
    return comments

# ---- MAIN LOOP ----
for label in ("positivos","negativos"):
    in_dir  = os.path.join(INPUT_DIR,  label)
    out_dir = os.path.join(OUTPUT_DIR, label)

    # iterate over each .hea in the folder
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(".hea"):
            continue
        basename = fn[:-4]
        record_path = os.path.join(in_dir, basename)

        # 1) load original record
        rec     = wfdb.rdrecord(record_path)
        sig     = rec.p_signal                # shape (n_samples, n_leads)
        orig_fs = rec.fs

        # 2) standardize
        sig_std = standardize_ecg(sig, orig_fs)  # (4000, n_leads)

        # 3) grab original metadata comments
        hea_in   = os.path.join(in_dir, basename + ".hea")
        comments = read_header_comments(hea_in)

        # 4) write out standardized record + preserved metadata
        wfdb.wrsamp(
            record_name=basename,
            fs=TARGET_FS,
            units=rec.units,               # e.g. ['mV']*12
            sig_name=rec.sig_name,         # e.g. ['I','II',...,'V6']
            p_signal=sig_std,              # (4000,12) physical units in mV
            fmt=rec.fmt,                   # typically ['16']*12
            adc_gain=[1000]*sig_std.shape[1],
            baseline=[0]*sig_std.shape[1],
            comments=comments,             # your original '# Age: …' etc.
            write_dir=out_dir
        )

        #print(f"Standardized → {label}/{basename}")

print("Done! New standardized dataset in:", OUTPUT_DIR)
