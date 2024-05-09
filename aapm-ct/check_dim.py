import h5py
import numpy as np 
import os
import gzip
import matplotlib.pyplot as plt


# Open the HDF5 file
# file = h5py.File('./real_data_hdf5/ground_truth_test/ground_truth_test_000.hdf5', 'r')
# file = h5py.File('./real_data_hdf5/observation_test/observation_test_027.hdf5', 'r')
# file = h5py.File('./real_data_hdf5/observation_test/observation_test_027.hdf5', 'r')

# file = h5py.File('./real_data_hdf5/observation_test/observation_test_027.hdf5', 'r')

# data = file["data"]

# print(data.shape)

# file.close()


sinogram = np.load(
    gzip.GzipFile(
        os.path.join("./real_data_modified/training_data", "Phantom_batch{}.npy.gz".format(1)),
        "r",
    )
)

print(sinogram.shape)

new = sinogram[0, :, :]

plt.figure(figsize=(10, 5))
plt.imshow(new, cmap='gray', aspect='auto')
plt.colorbar(label="Intensity")
plt.title(f"Sinogram {0}")
plt.xlabel("Detector Channels")
plt.ylabel("Projections")
plt.show()

