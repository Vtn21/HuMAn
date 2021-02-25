"""amass_to_tfrecord.py

Load the downloaded AMASS database (.npz files) into TFRecords.

This is the preferred binary file for TensorFlow, and has performance
advantages over loading the .npz files one by one.

This script can be executed just once after downloading and extracting
the AMASS dataset from https://amass.is.tue.mpg.de/. It uses the splits
recommended by the authors of the dataset.

A single TFRecord file is created for each AMASS sub-dataset. When loading
them for training or testing, they are easily joined together into the splits
using tf.data.TFRecordDataset.

Author: Victor T. N.
"""


import concurrent.futures
import glob
import numpy as np
import os


def check_amass(input_path, delete_no_pose):
    # Path to input files (.npz) with wildcards
    npz_path = os.path.join(input_path, "*/*.npz")
    # Iterate over all input files
    for npz_file in glob.iglob(npz_path):
        # Try to load specified file
        try:
            body_data = np.load(npz_file)
        except IOError:
            print(f"Error loading {npz_file}")
        except ValueError:
            print(f"allow_pickle=True required to load {npz_file}")
        else:
            if "poses" not in list(body_data.keys()):
                print(f"{npz_file} does not contain pose data")
                if delete_no_pose:
                    body_data.close()
                    try:
                        os.remove(npz_file)
                    except OSError:
                        print(f"Unable to remove {npz_file}")
                    else:
                        print(f"Deleted {npz_file}")
            elif not np.any(np.isfinite(body_data["poses"])):
                print(f"{npz_file} contains non-finite values")


if __name__ == "__main__":
    # Path to the datasets
    amass_home = "../../AMASS/datasets"
    # Split the sub-datasets into training, validation and testing
    # This names must match the subfolders of "amass_path"
    # Inexistent directories will be skipped
    amass_ds = ["KIT", "ACCAD", "BMLhandball", "BMLmovi", "BMLrub",
                "CMU", "DFaust_67", "EKUT", "Eyes_Japan_Dataset",
                "MPI_Limits", "TotalCapture", "HumanEva", "MPI_HDM05",
                "MPI_mosh", "SFU", "SSM_synced", "Transitions_mocap"]
    # Create a multiprocessing executor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Iterate over the sub-datasets
        for ds in amass_ds:
            # Create the input path
            input_path = os.path.join(amass_home, ds)
            # Start the process
            executor.submit(check_amass, input_path, True)
