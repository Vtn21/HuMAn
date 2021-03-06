"""npz2mat.py

Author: Victor T. N.
"""


import glob
import numpy as np
from scipy.io import savemat


if __name__ == "__main__":
    # Retrieve all recordings from subfolders
    for rec in glob.iglob("*/*.npz"):
        # Load data (dict of NumPy arrays)
        data = np.load(rec)
        # Retrieve the file name
        name = rec.split(".")[0] + ".mat"
        # Save to MATLAB file (.mat)
        savemat(name, data)
