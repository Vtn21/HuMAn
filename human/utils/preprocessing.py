"""preprocessing.py

Author: Victor T. N.
"""


import glob
import numpy as np
import os
import tqdm
from human.utils import features
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def amass_to_tfrecord(input_directory, output_tfrecord, framerate_drop=[1],
                      window_size=200, window_stride=50, max_betas=10,
                      tqdm_desc="Dataset (split)", tqdm_pos=0):
    # Path to all input files (.npz) from this sub-dataset
    npz_list = glob.glob(os.path.join(input_directory, "*/*.npz"))
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        # Iterate over all input files
        for npz_file in tqdm.tqdm(npz_list, total=len(npz_list),
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
                    if str(body_data["gender"]) == "male":
                        gender_int = -1
                    elif str(body_data["gender"]) == "female":
                        gender_int = 1
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
                            body_data["betas"][:num_betas])
                    }
                    # Data augmentation: framerate drop
                    for fr_drop in framerate_drop:
                        # Fill the "dt" field in the feature dict
                        feature["dt"] = features._float_feature(
                            [fr_drop/body_data["mocap_framerate"]])
                        # Keep only 24 first joints (72 angles), which discards
                        # hands and follows the STAR model definition, while
                        # also skipping frames to simulate a lower framerate
                        poses = body_data["poses"][::fr_drop, :72]
                        # Discard recordings shorter that the window size
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