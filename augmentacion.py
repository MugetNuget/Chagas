import os
import wfdb
import numpy as np
import random
import shutil

# --- Configuration ---
INPUT_FOLDER = "dataset_submuestreado\positivos"  # Folder with your .hea and .dat files
OUTPUT_FOLDER = "dataset_submuestreado\augmented_signals" # Folder to save augmented signals
TARGET_TOTAL_SIGNALS = 3000

# Augmentation parameters (adjust as needed)
NOISE_LEVEL_RANGE = (0.005, 0.02) # Percentage of signal's std deviation
SCALING_FACTOR_RANGE = (0.9, 1.1) # Multiplicative factor
TIME_SHIFT_SAMPLES_RANGE = (-10, 10) # Max samples to shift left/right (integer)

# --- Helper Augmentation Functions ---

def add_noise(signal, noise_level_range):
    """Adds Gaussian noise to the signal."""
    if signal.ndim == 1:
        signal = signal[:, np.newaxis] # Ensure 2D for consistent processing

    augmented_signal = signal.copy()
    for i in range(signal.shape[1]): # Iterate over channels
        channel_signal = signal[:, i]
        noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
        noise = np.random.normal(0, np.std(channel_signal) * noise_level, channel_signal.shape)
        augmented_signal[:, i] = channel_signal + noise
    return augmented_signal

def scale_signal(signal, scale_factor_range):
    """Scales the signal amplitude."""
    scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])
    return signal * scale_factor

def time_shift_signal(signal, shift_range_samples):
    """Shifts the signal in time, padding with zeros."""
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    shift_amount = random.randint(shift_range_samples[0], shift_range_samples[1])
    augmented_signal = np.zeros_like(signal)

    if shift_amount == 0:
        return signal

    for i in range(signal.shape[1]): # Iterate over channels
        channel_signal = signal[:, i]
        if shift_amount > 0:  # Shift right, pad left
            augmented_signal[shift_amount:, i] = channel_signal[:-shift_amount]
        else:  # Shift left, pad right
            augmented_signal[:shift_amount, i] = channel_signal[-shift_amount:]
    return augmented_signal


def apply_random_augmentation(p_signal):
    """Applies one or more random augmentations."""
    augmented_signal = p_signal.copy()
    
    # Decide which augmentations to apply (can apply multiple)
    if random.random() < 0.7: # 70% chance to add noise
        augmented_signal = add_noise(augmented_signal, NOISE_LEVEL_RANGE)
    
    if random.random() < 0.7: # 70% chance to scale
        augmented_signal = scale_signal(augmented_signal, SCALING_FACTOR_RANGE)

    if random.random() < 0.5: # 50% chance to time shift
        # Only shift if signal is long enough for a meaningful shift
        if augmented_signal.shape[0] > abs(TIME_SHIFT_SAMPLES_RANGE[0]) * 2 and \
           augmented_signal.shape[0] > abs(TIME_SHIFT_SAMPLES_RANGE[1]) * 2 :
            augmented_signal = time_shift_signal(augmented_signal, TIME_SHIFT_SAMPLES_RANGE)
            
    return augmented_signal

# --- Main Script ---
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        return

    if os.path.exists(OUTPUT_FOLDER):
        print(f"Warning: Output folder '{OUTPUT_FOLDER}' already exists. It will be overwritten.")
        shutil.rmtree(OUTPUT_FOLDER) # Remove existing output folder
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: '{OUTPUT_FOLDER}'")

    # 1. Get list of original record names
    record_names = []
    for f_name in os.listdir(INPUT_FOLDER):
        if f_name.endswith(".hea"):
            record_names.append(os.path.splitext(f_name)[0])

    if not record_names:
        print(f"No .hea files found in '{INPUT_FOLDER}'.")
        return

    num_original_signals = len(record_names)
    print(f"Found {num_original_signals} original signals.")

    if num_original_signals == 0:
        print("No signals to process.")
        return
        
    if TARGET_TOTAL_SIGNALS <= num_original_signals:
        print(f"Target signals ({TARGET_TOTAL_SIGNALS}) is less than or equal to original signals ({num_original_signals}).")
        print("Copying original signals only.")
        for i, rec_name in enumerate(record_names):
            if i >= TARGET_TOTAL_SIGNALS:
                break
            try:
                # Read original
                p_signal, fields = wfdb.rdsamp(os.path.join(INPUT_FOLDER, rec_name))
                
                # Write original to output
                output_rec_path = os.path.join(OUTPUT_FOLDER, rec_name)
                wfdb.wrsamp(
                    record_name=output_rec_path,
                    fs=fields['fs'],
                    units=fields['units'],
                    sig_name=fields['sig_name'],
                    p_signal=p_signal,
                    fmt=fields['fmt'],
                    comments=fields.get('comments', []) + [f"Original signal, copied."]
                )
            except Exception as e:
                print(f"Error processing original {rec_name}: {e}")
        print(f"Copied {min(TARGET_TOTAL_SIGNALS, num_original_signals)} original signals to '{OUTPUT_FOLDER}'.")
        return


    # 2. Copy original signals to output folder first
    print("Copying original signals to output folder...")
    for rec_name in record_names:
        try:
            # Read original
            p_signal, fields = wfdb.rdsamp(os.path.join(INPUT_FOLDER, rec_name))
            
            # Write original to output
            # wfdb.wrsamp uses the record_name argument as the base for file names.
            # If record_name includes a path, it writes to that path.
            output_rec_path = os.path.join(OUTPUT_FOLDER, rec_name)
            
            wfdb.wrsamp(
                record_name=output_rec_path, # This will be <OUTPUT_FOLDER>/<rec_name>
                fs=fields['fs'],
                units=fields['units'],
                sig_name=fields['sig_name'],
                p_signal=p_signal,
                fmt=fields['fmt'], # Important to preserve original format if possible
                comments=fields.get('comments', []) + [f"Original signal, copied."]
            )
        except Exception as e:
            print(f"Error processing original {rec_name}: {e}")
            continue
            
    print(f"Copied {num_original_signals} original signals.")
    
    signals_generated_count = num_original_signals
    augmentations_needed = TARGET_TOTAL_SIGNALS - num_original_signals
    
    if augmentations_needed <= 0:
        print("Target already met by original signals. No augmentation needed.")
        return

    print(f"Starting augmentation to generate {augmentations_needed} more signals...")

    aug_iteration = 0
    while signals_generated_count < TARGET_TOTAL_SIGNALS:
        # Cycle through original signals for augmentation
        original_rec_idx = aug_iteration % num_original_signals
        original_rec_name = record_names[original_rec_idx]
        
        try:
            # Read original signal again for augmentation
            p_signal_orig, fields_orig = wfdb.rdsamp(os.path.join(INPUT_FOLDER, original_rec_name))

            # Apply augmentations
            augmented_p_signal = apply_random_augmentation(p_signal_orig)

            # Create new record name for the augmented signal
            # Suffix example: _aug_0, _aug_1, ... to ensure unique names overall
            aug_suffix = f"aug_{aug_iteration}"
            augmented_rec_name_base = f"{original_rec_name}_{aug_suffix}"
            augmented_rec_path = os.path.join(OUTPUT_FOLDER, augmented_rec_name_base)

            # Prepare comments
            new_comments = fields_orig.get('comments', []) + [f"Augmented from {original_rec_name} (iteration {aug_iteration})"]

            # Save augmented signal
            wfdb.wrsamp(
                record_name=augmented_rec_path,
                fs=fields_orig['fs'],
                units=fields_orig['units'],
                sig_name=fields_orig['sig_name'],
                p_signal=augmented_p_signal,
                fmt=fields_orig['fmt'],
                comments=new_comments
            )
            
            signals_generated_count += 1
            aug_iteration += 1

            if signals_generated_count % 100 == 0: # Progress update
                print(f"Generated {signals_generated_count}/{TARGET_TOTAL_SIGNALS} signals...")

        except Exception as e:
            print(f"Error augmenting {original_rec_name} (iter {aug_iteration}): {e}")
            # To avoid infinite loop on a problematic file, increment aug_iteration anyway
            # or skip this file in future iterations for this augmentation run.
            # For simplicity, we'll just increment and hope other files work.
            aug_iteration += 1 
            if signals_generated_count >= TARGET_TOTAL_SIGNALS: # Check in case error happened on last one
                break
            continue # Move to next augmentation attempt

    print(f"--- Augmentation Complete ---")
    print(f"Total signals in '{OUTPUT_FOLDER}': {signals_generated_count}")
    print(f"(Originals: {num_original_signals}, Augmented: {signals_generated_count - num_original_signals})")

if __name__ == "__main__":
    # --- Create dummy input files for testing ---
    # You should replace this with your actual data setup
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Creating dummy folder '{INPUT_FOLDER}' for testing.")
        # Create a few dummy .hea and .dat files
        for i in range(5): # Create 5 dummy signals for testing
            record_name = f"test_sig_{i}"
            fs = 100
            sig_len = 1000
            n_sig = 2 # Number of channels
            
            # Dummy signal data (sine waves)
            t = np.linspace(0, sig_len/fs, sig_len, endpoint=False)
            dummy_p_signal = np.zeros((sig_len, n_sig))
            dummy_p_signal[:,0] = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(sig_len) # 5 Hz
            dummy_p_signal[:,1] = 0.5 * np.cos(2 * np.pi * 10 * t) + 0.3 * np.random.randn(sig_len) # 10 Hz
            
            units = ['mV', 'mV']
            sig_names = ['ECG1', 'ECG2']
            fmt = ['16', '16'] # e.g., 16-bit format

            # Save dummy signal using wfdb.wrsamp
            # Need to save to INPUT_FOLDER
            full_rec_path = os.path.join(INPUT_FOLDER, record_name)
            wfdb.wrsamp(record_name=full_rec_path,
                        fs=fs,
                        units=units,
                        sig_name=sig_names,
                        p_signal=dummy_p_signal,
                        fmt=fmt,
                        comments=["This is a dummy test signal."])
        print(f"Created 5 dummy signals in '{INPUT_FOLDER}'.")
        print("Please replace with your actual data or adjust TARGET_TOTAL_SIGNALS for testing.")
        print("For a real run, comment out or remove the dummy file creation block.")
        # Adjust target for the dummy data if you want a quick test
        # TARGET_TOTAL_SIGNALS = 15 # e.g., for 5 originals, this makes 10 augmentations
    # --- End of dummy file creation ---

    main()