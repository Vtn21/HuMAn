"""
view_tfrecord.py

Visualize the AMASS TFRecords created with "amass_to_tfrecord.py".
This script is important for showing if the data loaded into the
    TFRecord files still represent the human motion recordings.

Author: Victor T. N.
"""

import glob
import os
import time
import trimesh
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from star.tf.star import STAR  # noqa: E402


FACE_COLORS = (0, 255, 65, 255)  # Color for mesh faces
BG_COLORS = (116, 120, 128, 255)  # Color for background
FPS = 10


def parse_record(record):
    """Parse a single record from the dataset.

    Args:
        record (raw dataset record): a single record from the dataset.

    Returns:
        (parsed record): a single parsed record from the dataset.
    """
    num_names = {
        "num_poses": tf.io.FixedLenFeature([], tf.int64),
        "num_betas": tf.io.FixedLenFeature([], tf.int64)
    }
    num = tf.io.parse_single_example(record, num_names)
    feature_names = {
        "poses": tf.io.FixedLenFeature([num["num_poses"]*69], tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "betas": tf.io.FixedLenFeature([num["num_betas"]], tf.float32),
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
        poses (tensor): N x 69 tensor with the sequence of poses.
        betas (tensor): 1D tensor with all shape primitives.
        dt (tensor): float tensor with the time step for this sequence.
        gender (string): gender of the subject.
    """
    poses = tf.reshape(parsed_record["poses"], [-1, 69])
    betas = tf.reshape(parsed_record["betas"], [1, -1])
    dt = parsed_record["dt"]
    if parsed_record["gender"] == 1:
        gender = "female"
    elif parsed_record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, betas, dt, gender


def update_scene(scene):
    fId = next(frames)
    # Create a body with the STAR model
    body = star(poses_full[fId:fId+1, :], betas, tf.zeros((1, 3)))
    # Generate corresponding mesh
    body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                face_colors=FACE_COLORS)
    # Delete the previous mesh from the scene
    scene.delete_geometry("geometry_0")
    # Add the current mesh to the scene
    scene.add_geometry(body_mesh)


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Data splits (same as the filenames)
    splits = ["train", "valid", "test"]
    for split in splits:
        # Display information
        print(f"Loading \"{split}\" split\n")
        # Full path to the datasets of a specific split, with wildcards
        tfr_paths = os.path.join(tfr_home, split, "*.tfrecord")
        # Expand with glob
        tfr_list = glob.glob(tfr_paths)
        # Load the TFRecords as a Dataset
        dataset = tf.data.TFRecordDataset(tfr_list,
                                          num_parallel_reads=os.cpu_count())
        # Shuffle the dataset
        dataset = dataset.shuffle(1000,
                                  reshuffle_each_iteration=True)
        # Take a record as example
        record = next(iter(dataset))
        # Parse and decode
        poses, betas, dt, gender = decode_record(parse_record(record))
        # Complete the poses tensor with zeros
        poses_zeros = tf.zeros((poses.shape[0], 3))
        poses_full = tf.concat((poses_zeros, poses), 1)
        # Path to the STAR model
        path_model = os.path.join("../../AMASS/models/star", gender + ".npz")
        # Create STAR model
        star = STAR(path_model=path_model, num_betas=betas.shape[1])
        # Create iterator for the frame number
        frames = iter(range(0, poses.shape[0], int(1/(FPS * dt))))
        fId = next(frames)
        # Create a body with the STAR model
        body = star(poses_full[fId:fId+1, :], betas, tf.zeros((1, 3)))
        # Generate corresponding mesh
        body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                    face_colors=FACE_COLORS)
        # Create a scene with this mesh
        scene = trimesh.Scene(body_mesh)
        # Create a viewer
        try:
            scene.show(callback=update_scene,
                       smooth=False,
                       background=BG_COLORS,
                       resolution=(800, 600),
                       callback_period=1/FPS)
        except StopIteration:
            time.sleep(1)
