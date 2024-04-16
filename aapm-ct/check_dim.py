import h5py
import numpy as np 
import os
import gzip

# Open the HDF5 file
# file = h5py.File('./real_data_hdf5/ground_truth_test/ground_truth_test_000.hdf5', 'r')
file = h5py.File('./real_data_hdf5/observation_test/observation_test_027.hdf5', 'r')

data = file["data"]

print(data.shape)

file.close()


sinogram = np.load(
    gzip.GzipFile(
        os.path.join("./data/", "Sinogram_batch{}.npy.gz".format(1)),
        "r",
    )
)

print(sinogram.shape)