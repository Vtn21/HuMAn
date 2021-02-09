"""
amass_to_tfrecord.py

Load the downloaded AMASS database (.npz files) into TFRecords.
This is the preferred binary file for TensorFlow, and has performance
    advantages over loading the .npz files one by one.
Also, TFRecords are preferred over the tf.data.Dataset, as AMASS is large
    (23 GB of .npz files) and won't fit on RAM (at least not on my computer :D)

Author: Victor T. N.
"""

import concurrent.futures
import glob
import numpy as np
import os
import sys
import tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf


def _bytes_feature(value):
    """BytesList convertor for TFRecord.

    Args:
        value (string / byte): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to BytesList
    """
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """FloatList convertor for TFRecord.

    Args:
        value (float32 / float64): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to FloatList
    """
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Int64List convertor for TFRecord.

    Args:
        value (bool / enum / (u)int32 / (u)int64): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to Int64List
    """
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def amass_example(bdata, framerate_drop=1, max_betas=10):
    """Create a TF Example from an AMASS recording.
    There is the option to drop down the framerate (augmentation).

    Args:
        bdata (numpy dict): uncompressed numpy data from the recording.
        framerate_drop (int, optional): Used to drop the original framerate
                                        (data augmentation). Defaults to 1.
        max_betas (int, optional): Limits the maximum number of shape
                                   components. Defaults to 10.

    Returns:
        [tf.train.Example]: relevant data fields wrapped into TF Example.
    """
    # Turning string gender into integer
    if str(bdata["gender"]) == "male":
        gender_int = -1
    elif str(bdata["gender"]) == "female":
        gender_int = 1
    else:
        gender_int = 0
    # Ensure that the number of betas is not greater than what is available
    num_betas = min(max_betas, bdata["betas"].shape[0])
    # Keep only joint poses, discarding body orientation
    # framerate_drop acts here, picking just part of the array
    poses = bdata["poses"][0::framerate_drop, 3:72]
    # Store the length (used for parsing)
    num_poses = poses.shape[0]
    # Build feature dict
    feature = {
        # Flattened poses array
        "poses": _float_feature(poses.flatten()),
        # Number of pose frames
        "num_poses": _int64_feature([num_poses]),
        # Shape components (betas), up to the maximum defined amount
        "betas": _float_feature(bdata["betas"][:num_betas]),
        # Store the number of betas
        "num_betas": _int64_feature([num_betas]),
        # Time interval between recordings, considering framerate drop
        "dt": _float_feature([framerate_drop/bdata["mocap_framerate"]]),
        # Gender encoded into integer
        "gender": _int64_feature([gender_int])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(input_path, output_path, description, position):
    """Parses a full sub-dataset from AMASS and writes to a TFRecord file.
    Each sub-dataset is handled by a separate process.
    Files can be joined when loading with tf.data.TFRecordDataset.

    Args:
        input_path (string): path to the sub-dataset, containing .npz files.
        output_path (string): full path to the output file, with extension.
        description (string): description for the tqdm progress bar.
        position (int): position for the tqdm progress bar.
    """
    # Path to input files (.npz) with wildcards
    npz_path = os.path.join(input_path, "*/*.npz")
    # Expand with glob
    npz_list = glob.glob(npz_path)
    # Create a record writer
    with tf.io.TFRecordWriter(output_path) as writer:
        # Iterate over all input files
        for i in tqdm.trange(len(npz_list), desc=description,
                             position=position, dynamic_ncols=True,
                             mininterval=0.5):
            # Try to load specified file
            try:
                bdata = np.load(npz_list[i])
            except IOError:
                print("Error loading " + npz_list[i])
            except ValueError:
                print("allow_pickle=True required to load " + npz_list[i])
            else:
                if "poses" not in list(bdata.keys()):
                    # Skip a non-valid file
                    continue
                else:
                    for fr_drop in [1, 2, 4, 10]:
                        # Create the Example
                        tf_example = amass_example(bdata, fr_drop)
                        # Write to TFRecord file
                        writer.write(tf_example.SerializeToString())


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
                                output_path, description, position)
                # Increment position counter
                position += 1
