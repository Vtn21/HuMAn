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
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb


def extract_stats(input_path):
    # Path to input files (.npz) with wildcards
    npz_path = os.path.join(input_path, "*/*.npz")
    # Expand with glob
    npz_list = glob.glob(npz_path)
    # Lists to store stats
    stats = {"length": [], "framerate": []}
    # Iterate over all input files
    for npz_file in npz_list:
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
            else:
                stats["length"].append(body_data["poses"].shape[0])
                stats["framerate"].append(body_data["mocap_framerate"].item())
    return stats


if __name__ == "__main__":
    # Path to the datasets
    amass_home = "../../AMASS/datasets"
    # Sub-datasets
    # This names must match the subfolders of "amass_path"
    amass_ds = ["KIT", "ACCAD", "BMLhandball", "BMLmovi", "BMLrub",
                "CMU", "DFaust_67", "EKUT", "Eyes_Japan_Dataset",
                "MPI_Limits", "TotalCapture", "HumanEva", "MPI_HDM05",
                "MPI_mosh", "SFU", "SSM_synced", "Transitions_mocap"]
    input_path = [os.path.join(amass_home, ds) for ds in amass_ds]
    # Start the process
    with concurrent.futures.ProcessPoolExecutor() as executor:
        stats = executor.map(extract_stats, input_path)
    lengths = []
    framerates = []
    for stat in stats:
        lengths.append(np.array(stat["length"]))
        framerates.append(np.array(stat["framerate"]))
    result = {"length": np.concatenate(lengths),
              "framerate": np.concatenate(framerates)}
    print(f"Minimum length: {np.min(result['length'])} frames")
    print(f"Maximum length: {np.max(result['length'])} frames")
    print(f"Lengths: {np.unique(result['length'])}")
    print(f"Framerates: {np.unique(result['framerate'])}")
    # Plot histograms
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    for i, key in enumerate(result.keys()):
        axs[i].hist(result[key], bins=20)
        axs[i].set_title(key)
    plt.show()
    pdb.set_trace()
