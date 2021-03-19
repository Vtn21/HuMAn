"""preprocess_amass.py"

Uses the "amass_to_tfrecord" preprocessing function to create TFRecord files
from AMASS .npz files. This functions also handles preprocessing and data
augmentation, as well as converting recording to fixed-length sequences.

Author: Victor T. N.
"""


import concurrent.futures
import os
import sys
from human.utils.preprocessing import amass_to_tfrecord


# Use the following framerate drop rates to augment data
FRAMERATE_DROP = [1, 2, 4, 5]
SEQ_LENGTH = 1024


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
    # Iterate over all splits, to create subdirectories if needed
    for split in amass_splits.keys():
        # Path for the corresponding subdirectory
        tfr_subdir = os.path.join(tfr_home, split + f"_{SEQ_LENGTH}")
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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # This position variable controls the position of progress bars
        tqdm_pos = 0
        # Iterate over all splits
        for split in amass_splits.keys():
            # Iterate over the sub-datasets
            for sub_ds in amass_splits[split]:
                # Create the input path
                input_directory = os.path.join(amass_home, sub_ds)
                # Create the output path
                output_tfrecord = os.path.join(
                    tfr_home, split + f"_{SEQ_LENGTH}", sub_ds + ".tfrecord")
                # Create a description string
                tqdm_desc = sub_ds + " (" + split + ")"
                # Start the process
                executor.submit(amass_to_tfrecord,
                                input_directory=input_directory,
                                output_tfrecord=output_tfrecord,
                                framerate_drop=FRAMERATE_DROP,
                                seq_length=SEQ_LENGTH,
                                tqdm_desc=tqdm_desc, tqdm_pos=tqdm_pos)
                # Increment position counter
                tqdm_pos += 1
