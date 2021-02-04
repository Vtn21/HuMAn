"""
load_amass.py

Load the downloaded AMASS database (.npz files) into TFRecords.
This is the preferred binary file for TensorFlow, and has performance
    advantages over loading the .npz files one by one.
Also, TFRecords are preferred over the tf.data.Dataset, as AMASS is large
    (23 GB of .npz files) and won't fit on RAM (at least not on my computer :D)

Author: Victor T. N.
"""

import glob
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf


if __name__ == "__main__":
    # Path to the datasets
    amass_path = "../../AMASS/datasets"
    # Split the sub-datasets into training, validation and testing
    # This names must match the subfolders of "amass_path"
    # Inexistent directories will be skipped
    amass_splits = {
        "train": ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU",
                  "DFaust_67", "EKUT", "Eyes_Japan_Dataset", "KIT",
                  "MPI_Limits", "TotalCapture"],
        "valid": ["HumanEva", "MPI_HDM05", "MPI_mosh", "SFU"],
        "test": ["SSM_synced", "Transitions_mocap"]
    }
    # Path to save the TFRecords files
    tfr_path = "../../AMASS/tfrecords"
    for split in amass_splits.keys():
        # Full path to the TFRecords file (same name as the respective split)
        tfr_file = os.path.join(tfr_path, "".join((split, ".tfrecord")))
        for sub_ds in amass_splits[split]:
            npz_glob = os.path.join(amass_path, sub_ds, "*/*.npz")
            for npz_file in glob.iglob(npz_glob):
                bdata = np.load(npz_file)
                if "poses" not in list(bdata.keys()):
                    continue
                else:
                    print("Recording: %s" % recording)
                    print('Data keys available:%s' % list(bdata.keys()))
                break
