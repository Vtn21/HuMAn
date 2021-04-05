"""preprocess_amass.py"

Uses the "amass_to_tfrecord" preprocessing function to create TFRecord files
from AMASS .npz files. This functions also handles preprocessing and data
augmentation, as well as converting recordings to fixed-length sequences.

Author: Victor T. N.
"""


import concurrent.futures
import glob
import numpy as np
import os
import sys
from human.utils.preprocessing import amass_to_tfrecord


# Use the following framerate drop rates to augment data
FRAMERATE_DROP = [1, 2, 4, 5]

# Create TFRecords with the following sequence lengths
SEQ_LENGTH = [256, 512, 1024]


if __name__ == "__main__":
    # Path to the datasets
    amass_home = "../../AMASS/datasets"
    # Path to save the TFRecords files
    tfr_home = "../../AMASS/tfrecords"
    # List to store call arguments
    input_npz_list = []
    sub_dir = []
    record_name = []
    # Sub-directory 1: general training data
    # Iterate through all sub-datasets
    for sub_ds in ["KIT", "ACCAD", "BMLmovi", "BMLrub", "CMU", "DFaust_67",
                   "EKUT", "Eyes_Japan_Dataset", "MPI_Limits", "TotalCapture",
                   "Transitions_mocap"]:
        # Append to lists
        input_npz_list.append(glob.glob(os.path.join(
            amass_home, sub_ds, "*/*.npz")))
        sub_dir.append("train")
        record_name.append(sub_ds)
    # Sub-directory 2: general validation data
    # Iterate through all sub-datasets
    for sub_ds in ["HumanEva", "MPI_mosh", "SFU", "SSM_synced"]:
        # Append to lists
        input_npz_list.append(glob.glob(os.path.join(
            amass_home, sub_ds, "*/*.npz")))
        sub_dir.append("valid")
        record_name.append(sub_ds)
    # Sub-directory 3: BMLhandball (motion-specific)
    # Iterate through all subjects
    for subject in os.listdir(os.path.join(amass_home, "BMLhandball")):
        # Append to lists
        input_npz_list.append(glob.glob(os.path.join(
            amass_home, "BMLhandball", subject, "*.npz")))
        sub_dir.append("BMLhandball")
        record_name.append(subject)
    # Sub-directory 4: MPI_HDM05 (subject-specific)
    # Iterate through all subjects
    for subject in os.listdir(os.path.join(amass_home, "MPI_HDM05")):
        train = []
        valid = []
        for motion_type in ["01", "02", "03", "04", "05", "06", "07", "08"]:
            # List all recordings for a specific motion type
            recs = glob.glob(os.path.join(amass_home, "MPI_HDM05", subject,
                                          f"HDM_{subject}_{motion_type}*.npz"))
            # Compute 10 % of all recordings (set aside for validation)
            num_valid = int(np.round(len(recs)/10))
            # Create training and validation splits
            if num_valid == 0:
                train += recs
            else:
                train += recs[:-num_valid]
                valid += recs[-num_valid:]
        # Append to lists
        input_npz_list.append(train)
        sub_dir.append("MPI_HDM05")
        record_name.append(f"{subject}_train")
        input_npz_list.append(valid)
        sub_dir.append("MPI_HDM05")
        record_name.append(f"{subject}_valid")
    # Create an executor, to run one process for each triad of call arguments
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # This position variable controls the position of progress bars
        tqdm_pos = 0
        # Iterate through the arguments lists
        for i in range(len(input_npz_list)):
            # Iterate through all sequence lengths
            for seq_len in SEQ_LENGTH:
                # Create the output directory
                output_dir = os.path.join(tfr_home, f"{sub_dir[i]}_{seq_len}")
                # Create the directory, if it doesn't exist
                try:
                    os.makedirs(output_dir)
                except FileExistsError:
                    print(f"Directory {output_dir} already exists "
                          "and will be used.")
                except FileNotFoundError:
                    print(f"Directory {output_dir} is invalid.")
                    sys.exit()
                else:
                    print(f"Creating new directory {output_dir}.")
                # Path for the output TFRecord
                output_tfrecord = os.path.join(
                    output_dir, record_name[i] + ".tfrecord")
                # Description for the progress bar
                desc = f"{sub_dir[i]}_{seq_len}/{record_name[i]}.tfrecord"
                # Spawn a process
                executor.submit(amass_to_tfrecord,
                                input_npz_list=input_npz_list[i],
                                output_tfrecord=output_tfrecord,
                                framerate_drop=FRAMERATE_DROP,
                                seq_length=seq_len,
                                tqdm_desc=desc,
                                tqdm_pos=tqdm_pos)
                # Increment position counter
                tqdm_pos += 1
