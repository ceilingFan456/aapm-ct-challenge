import os
import numpy as np
import gzip

def load_data(path, batch_num):
    sinogram_path = os.path.join(path, f"Sinogram_batch{batch_num}.npy.gz")
    fbp_path = os.path.join(path, f"FBP128_batch{batch_num}.npy.gz")
    phantom_path = os.path.join(path, f"Phantom_batch{batch_num}.npy.gz")

    with gzip.GzipFile(sinogram_path, 'r') as f:
        sinogram = np.load(f)
    with gzip.GzipFile(fbp_path, 'r') as f:
        fbp = np.load(f)
    with gzip.GzipFile(phantom_path, 'r') as f:
        phantom = np.load(f)

    return sinogram, fbp, phantom

def save_data(data, path, filename):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    with gzip.GzipFile(filepath, 'w') as f:
        np.save(f, data)

def split_and_save_data(path, output_path):
    all_sinograms = []
    all_fbps = []
    all_phantoms = []

    # Load data from all batches
    for batch_num in range(1, 5):
        sinogram, fbp, phantom = load_data(path, batch_num)
        all_sinograms.append(sinogram)
        all_fbps.append(fbp)
        all_phantoms.append(phantom)

    # Concatenate all batches
    all_sinograms = np.concatenate(all_sinograms, axis=0)
    all_fbps = np.concatenate(all_fbps, axis=0)
    all_phantoms = np.concatenate(all_phantoms, axis=0)

    # Split data
    train_sinograms = all_sinograms[:2800]
    val_sinograms = all_sinograms[2800:3600]
    test_sinograms = all_sinograms[3600:4000]

    train_fbps = all_fbps[:2800]
    val_fbps = all_fbps[2800:3600]
    test_fbps = all_fbps[3600:4000]

    train_phantoms = all_phantoms[:2800]
    val_phantoms = all_phantoms[2800:3600]
    test_phantoms = all_phantoms[3600:4000]

    # Save data into the respective folders
    datasets = {
        'training_data': (train_sinograms, train_fbps, train_phantoms),
        'validation_data': (val_sinograms, val_fbps, val_phantoms),
        'test_data': (test_sinograms, test_fbps, test_phantoms),
    }

    for dataset, (sinograms, fbps, phantoms) in datasets.items():
        for batch_num in range(1, 5):
            s = sinograms.shape[0] // 4
            start_idx = (batch_num - 1) * s
            end_idx = batch_num * s
            batch_sinograms = sinograms[start_idx:end_idx]

            s = fbps.shape[0] // 4
            start_idx = (batch_num - 1) * s
            end_idx = batch_num * s            
            batch_fbps = fbps[start_idx:end_idx]

            s = phantoms.shape[0] // 4
            start_idx = (batch_num - 1) * s
            end_idx = batch_num * s
            batch_phantoms = phantoms[start_idx:end_idx]

            save_data(batch_sinograms, os.path.join(output_path, dataset), f"Sinogram_batch{batch_num}.npy.gz")
            save_data(batch_fbps, os.path.join(output_path, dataset), f"FBP128_batch{batch_num}.npy.gz")
            save_data(batch_phantoms, os.path.join(output_path, dataset), f"Phantom_batch{batch_num}.npy.gz")

# Paths
input_path = "data"
output_path = "raw_data"

# Split and save the data
split_and_save_data(input_path, output_path)

