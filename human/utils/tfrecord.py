import glob
import numpy as np
import os
import tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


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


def amass_to_example(body_data, framerate_drop=1, max_betas=10):
    """Create a TF Example from an AMASS recording.
    There is the option to drop down the framerate (augmentation).

    Args:
        body_data (numpy dict): uncompressed numpy data from the recording.
        framerate_drop (int, optional): Used to drop the original framerate
                                        (data augmentation). Defaults to 1.
        max_betas (int, optional): Limits the maximum number of shape
                                   components. Defaults to 10.

    Returns:
        [tf.train.Example]: relevant data fields wrapped into TF Example.
    """
    # Turning string gender into integer
    if str(body_data["gender"]) == "male":
        gender_int = -1
    elif str(body_data["gender"]) == "female":
        gender_int = 1
    else:
        gender_int = 0
    # Ensure that the number of betas is not greater than what is available
    num_betas = min(max_betas, body_data["betas"].shape[0])
    # Keep the first 24 joints (72 angles), which discards hands
    # framerate_drop acts here, picking just part of the array
    poses = body_data["poses"][::framerate_drop, :72]
    # Build feature dict
    feature = {
        # Flattened poses array
        "poses": _float_feature(poses.flatten()),
        # Shape components (betas), up to the maximum defined amount
        "betas": _float_feature(body_data["betas"][:num_betas]),
        # Time interval between recordings, considering framerate drop
        "dt": _float_feature([framerate_drop/body_data["mocap_framerate"]]),
        # Gender encoded into integer
        "gender": _int64_feature([gender_int])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(input_path, output_path, framerate_drop, description,
                   position):
    """Parses a full sub-dataset from AMASS and writes to a TFRecord file.
    Each sub-dataset can be handled by a separate process.
    Files can be joined when loading with tf.data.TFRecordDataset.

    Args:
        input_path (string): path to the sub-dataset, containing .npz files.
        output_path (string): full path to the output file, with extension.
        framerate_drop (list): list of integers for dropping the original
                               framerate.
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
                body_data = np.load(npz_list[i])
            except IOError:
                print("Error loading " + npz_list[i])
            except ValueError:
                print("allow_pickle=True required to load " + npz_list[i])
            else:
                if "poses" not in list(body_data.keys()):
                    # Skip a non-valid file, silently
                    continue
                else:
                    for fr_drop in framerate_drop:
                        # Create the Example
                        tf_example = amass_to_example(body_data, fr_drop)
                        # Write to TFRecord file
                        writer.write(tf_example.SerializeToString())


def parse_record(record):
    """Parse a single record from the dataset.

    Args:
        record (raw dataset record): a single record from the dataset.

    Returns:
        (parsed record): a single parsed record from the dataset.
    """
    feature_names = {
        "poses": tf.io.VarLenFeature(tf.float32),
        "betas": tf.io.VarLenFeature(tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "gender": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


def decode_record(parsed_record):
    """Decode a previously parsed record.

    Args:
        parsed_record (raw dataset record): a single record from the dataset,
                                            previously parsed by the
                                            "parse_record" function.

    Returns:
        poses (tensor): N x 72 tensor with the sequence of poses.
        betas (tensor): 1D tensor with all shape primitives.
        dt (tensor): float tensor with the time step for this sequence.
        gender (string): gender of the subject.
    """
    poses = tf.reshape(tf.sparse.to_dense(parsed_record["poses"]), [-1, 72])
    betas = tf.reshape(tf.sparse.to_dense(parsed_record["betas"]), [1, -1])
    dt = parsed_record["dt"]
    if parsed_record["gender"] == 1:
        gender = "female"
    elif parsed_record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, betas, dt, gender
