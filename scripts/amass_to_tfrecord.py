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
import os
import sys
from human.utils.tfrecord import write_tfrecord


# Save each recording with the following framerate drop rates
FRAMERATE_DROP = [1, 2, 4, 10]


if __name__ == "__main__":
    # Path to the datasets
    amass_home = "../../AMASS/datasets"
    # Split the sub-datasets into training, validation and testing
    # This names must match the subfolders of "amass_path"
    # Inexistent directories will be skipped
    amass_splits = {
        "train": ["KIT", "ACCAD", "BMLhandball", "BMLmovi", "BMLrub",
                  "CMU", "DFaust_67", "EKUT", "Eyes_Japan_Dataset",
                  "MPI_Limits", "TotalCapture"],
        "valid": ["HumanEva", "MPI_HDM05", "MPI_mosh", "SFU"],
        "test": ["SSM_synced", "Transitions_mocap"]
    }
    # Path to save the TFRecords files
    tfr_home = "../../AMASS/tfrecords"
    # Iterate over all splits
    for split in amass_splits.keys():
        # Path for the corresponding subdirectory
        tfr_subdir = os.path.join(tfr_home, split)
        # Create the subdirectory, if it doesn't exist
        try:
            os.mkdir(tfr_subdir)
        except FileExistsError:
            print(f"Directory {tfr_subdir} already exists and will be used.")
        except FileNotFoundError:
            print(f"Directory {tfr_subdir} is invalid.")
            sys.exit()
        else:
            print(f"Creating new directory {tfr_subdir}.")
    # Create a multiprocessing executor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # This position variable controls the position of progress bars
        position = 0
        # Iterate over all splits
        for split in amass_splits.keys():
            # Iterate over the sub-datasets
            for sub_ds in amass_splits[split]:
                # Create the input path
                input_path = os.path.join(amass_home, sub_ds)
                # Create the output path
                output_path = os.path.join(tfr_home, split,
                                           sub_ds + ".tfrecord")
                # Create a description string
                description = sub_ds + " (" + split + ")"
                # Start the process
                executor.submit(write_tfrecord, input_path,
                                output_path, FRAMERATE_DROP,
                                description, position)
                # Increment position counter
                position += 1
