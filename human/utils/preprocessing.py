"""preprocessing.py

Contains a single function for reading the .npz files from the original
AMASS dataset, preprocessing them and saving into TFRecords files.

This function is intended to process a single sub-dataset at a time,
making it suitable for multiprocessing.

See the "preprocess_amass.py" script for example usage.

Author: Victor T. N.
"""


import numpy as np
import os
import tqdm
from human.utils import features
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def amass_to_tfrecord(input_npz_list, output_tfrecord, framerate_drop=[1],
                      seq_length=256, max_horizon=0.5, percent_stride=0.25,
                      max_betas=10, tqdm_desc="", tqdm_pos=0):
    """Preprocesses and saves a set of AMASS recordings (from .npz files)
    into a TFRecord file. This function is suitable for multiprocessing.

    Args:
        input_npz_list (list): list of all .npz files (path strings) to be
            included in the output TFRecord file.
        output_tfrecord (string): full path to the output TFRecord file, that
            will contain the whole sub-dataset. It is recommended to use the
            .tfrecord extension to help identify the file type.
        framerate_drop (list, optional): a list of integers to perform data
            augmentation, by artificially creating recordings with lower
            framerates. Defaults to [1].
        seq_length (int, optional): desired sequence length to be used during
            training (a constant length makes training computationally more
            efficient). Defaults to 256.
        max_horizon (float, optional): maximum prediction horizon, in seconds.
            This number is used in conjunction with "seq_length" to create the
            complete window size. Defaults to 0.5 (twice the time of human
            reaction to tactile stimuli).
        percent_stride (float, optional): percentage of the maximum window size
            (defined dynamically using "seq_length" and "max_horizon") to
            define the stride size. Smaller percentages create larger datasets,
            with the cost of data redundancy. Defaults to 0.25.
        max_betas (int, optional): maximum number of shape components to be
            recorded. Defaults to 10.
        tqdm_desc (str, optional): description string to show in front of the
            tqdm progress bar. Defaults to "" (empty string).
        tqdm_pos (int, optional): position of the progress bar. It is
            recommended to increment this parameter in steps of 1 every time a
            new process is spawned. Defaults to 0.
    """
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        # Iterate over all input files
        for npz_file in tqdm.tqdm(input_npz_list, total=len(input_npz_list),
                                  desc=tqdm_desc, position=tqdm_pos,
                                  dynamic_ncols=True, mininterval=0.5):
            # Try to load the .npz file
            try:
                body_data = np.load(npz_file)
            except IOError:
                print(f"Error loading {npz_file}")
            except ValueError:
                print(f"allow_pickle=True required to load {npz_file}")
            else:
                # Remove files that do not contain pose data
                if "poses" not in list(body_data.keys()):
                    print(f"{npz_file} does not contain pose data")
                    body_data.close()
                    try:
                        os.remove(npz_file)
                    except OSError:
                        print(f"Unable to remove {npz_file}")
                    else:
                        print(f"Successfully removed {npz_file}")
                # Warn against files with invalid data
                elif not np.any(np.isfinite(body_data["poses"])):
                    print(f"{npz_file} contains invalid pose data")
                # File can be used
                else:
                    # Turn gender string into integer
                    if "female" in str(body_data["gender"]):
                        gender_int = 1
                    elif "male" in str(body_data["gender"]):
                        gender_int = -1
                    else:
                        gender_int = 0
                    # Ensure that the requested number of betas is not greater
                    # than what is available
                    num_betas = min(max_betas, body_data["betas"].shape[0])
                    # Build the immutable part of the feature dict
                    feature = {
                        # Gender encoded into integer
                        "gender": features._int64_feature([gender_int]),
                        # Shape components (betas)
                        "betas": features._float_feature(
                            body_data["betas"][:num_betas]),
                        # The sequence length
                        "seq_length": features._int64_feature([seq_length])
                    }
                    # Data augmentation: framerate drop
                    for fr_drop in framerate_drop:
                        # Fill the "dt" field in the feature dict
                        dt = fr_drop/body_data["mocap_framerate"]
                        feature["dt"] = features._float_feature([dt])
                        # Keep only 24 first joints (72 angles), which discards
                        # hands and follows the STAR model definition, while
                        # also skipping frames to simulate a lower framerate
                        poses = body_data["poses"][::fr_drop, :72]
                        # Calculate the actual window size, composed by a fixed
                        # number of frames defined by seq_length and a variable
                        # extra length to enable multiple prediction horizons
                        window_size = int(seq_length + max_horizon/dt)
                        # Define a stride for this specific window size
                        window_stride = int(window_size*percent_stride)
                        # Discard recordings shorter than the window size
                        if poses.shape[0] < window_size:
                            continue
                        else:
                            # Search the maximum length defined as
                            # window_size + N * window_stride (N natural)
                            # that fits inside the recording
                            target_size = window_size
                            while target_size + window_stride < poses.shape[0]:
                                target_size += window_stride
                            # Remove excessive frames from beginning and end
                            split_size = (poses.shape[0] - target_size)/2
                            poses = poses[int(np.ceil(split_size)):
                                          int(-np.floor(split_size))]
                            # Poses length now matches target_size, enabling
                            # extraction of windows
                            start = 0
                            while start + window_size <= poses.shape[0]:
                                poses_window = poses[start:start+window_size]
                                # Finish the feature definition
                                feature["poses"] = features._float_feature(
                                    poses_window.flatten())
                                # Create the example
                                tf_example = tf.train.Example(
                                    features=tf.train.Features(
                                        feature=feature))
                                # Write to TFRecord file
                                writer.write(tf_example.SerializeToString())
                                # Increment "start" to get a new window
                                start += window_stride


def amass_to_tfrecord_simple(input_npz, output_tfrecord,
                             max_horizon=0.5, max_betas=10):
    """Preprocesses and saves a single AMASS recording (a .npz file)
    into a TFRecord file.

    Args:
        input_npz (string): .npz file (path strings) to be included in the
            output TFRecord file.
        output_tfrecord (string): full path to the output TFRecord file, that
            will contain the selected recording. It is recommended to use the
            .tfrecord extension to help identify the file type.
        max_horizon (float, optional): maximum prediction horizon, in seconds.
            Defaults to 0.5 (twice the time of human reaction to tactile
            stimuli).
        max_betas (int, optional): maximum number of shape components to be
            recorded. Defaults to 10.
    """
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        # Try to load the .npz file
        try:
            body_data = np.load(input_npz)
        except IOError:
            print(f"Error loading {input_npz}")
        except ValueError:
            print(f"allow_pickle=True required to load {input_npz}")
        else:
            # Warn against files with invalid data
            if not np.any(np.isfinite(body_data["poses"])):
                print(f"{input_npz} contains invalid pose data")
            # File can be used
            else:
                # Turn gender string into integer
                if "female" in str(body_data["gender"]):
                    gender_int = 1
                elif "male" in str(body_data["gender"]):
                    gender_int = -1
                else:
                    gender_int = 0
                # Ensure that the requested number of betas is not greater
                # than what is available
                num_betas = min(max_betas, body_data["betas"].shape[0])
                # Convert framerate into sampling time
                dt = 1/body_data["mocap_framerate"]
                # Build the feature dict
                feature = {
                    # Gender encoded into integer
                    "gender": features._int64_feature([gender_int]),
                    # Shape components (betas)
                    "betas": features._float_feature(
                        body_data["betas"][:num_betas]),
                    # The sequence length
                    "seq_length": features._int64_feature(
                        [int(body_data["poses"].shape[0] - max_horizon/dt)]),
                    # Framerate converted into sampling time
                    "dt": features._float_feature([dt]),
                    # The poses array
                    "poses": features._float_feature(
                        body_data["poses"][:, :72].flatten())
                }
                # Create the example
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature=feature))
                # Write to TFRecord file
                writer.write(tf_example.SerializeToString())
