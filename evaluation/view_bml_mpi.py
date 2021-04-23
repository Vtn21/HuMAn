"""view_bml_mpi.py

Simple script to print the results obtained when evaluating the model on the
BMLhandball and MPI-HDM05 datasets.

Author: Victor T. N.
"""


import numpy as np


if __name__ == '__main__':
    for folder in ["bmlhandball", "mpihdm05"]:
        for model in ["universal", "train", "transfer"]:
            for dataset in ["valid", folder[0:3]]:
                data = np.load(f"{folder}/{model}_{dataset}.npz")
                mean = np.round(data["mean"], 4)
                stdev = np.round(data["stdev"], 4)
                print(f"{folder}, {model}, {dataset}: "
                      f"mean={mean}, stdev={stdev}")
