import h5py
import numpy as np 
import os
import gzip

# Open the HDF5 file
# file = h5py.File('./real_data/ground_truth_test/ground_truth_test_000.hdf5', 'r')
# file = h5py.File('./real_data/observation_test/observation_test_027.hdf5', 'r')

folders = ["ground_truth_validation", "ground_truth_test", "observation_validation", "observation_test"]
raw_folder = "./real_data_hdf5"
output_folder = "./real_data_modified"

for folder in folders:
    data_list = []
    folder_path = os.path.join(raw_folder, folder)
    
    for id in [str(i).zfill(3) for i in range(10)]:
        name = f"{folder}_{id}"
        file_path = os.path.join(folder_path, f"{name}.hdf5")
        
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as file:
            dataset = file['data']
            data_list.append(np.array(dataset))
    
    data = np.concatenate(data_list, axis=0)

    ## store as training/ validation
    if folder.endswith("validation"):
        output_directory = os.path.join(output_folder, "training_data")
        os.makedirs(output_directory, exist_ok=True)

    else:
        output_directory = os.path.join(output_folder, "validation_data")
        os.makedirs(output_directory, exist_ok=True)


    ## store as phantom, FBP or sinogram
    if folder.startswith("ground_truth"):
        ## modify data to have the square -> circle-> square shape
        old_size = data.shape[-1]
        new_size = int(np.ceil(old_size * np.sqrt(2)))
        new_data = np.zeros((data.shape[0] // 100 , new_size, new_size))
        start_index = (new_size - old_size) // 2
        new_data[:, start_index: start_index + old_size, start_index: start_index + old_size] = data[data.shape[0] // 100, :, :]

        print(f"old_size={old_size}, new_size={new_size}, start_index={start_index}")
        print(f"old_shape={data.shape}, new_shape={new_data.shape}")

        output_path = os.path.join(output_directory, "Phantom_batch1.npy.gz")
        with gzip.open(output_path, 'wb') as f:
            np.save(f, new_data)

        output_path = os.path.join(output_directory, "FBP128_batch1.npy.gz")
        with gzip.open(output_path, 'wb') as f:
            np.save(f, new_data)

    elif folder.startswith("observation"):   
        output_path = os.path.join(output_directory, "Sinogram_batch1.npy.gz")
        with gzip.open(output_path, 'wb') as f:
            np.save(f, data[data.shape[0] //100, :, :])
