"""visualization.py

Contains tools for visualizing AMASS recordings, directly from .npz files or
from generated TFRecords.

The recommended way of viewing an AMASS (.npz) recording is by using the
"view_amass_npz" function, while a TFRecord file (created after preprocessing)
should be visualized using "view_tfrecord".

Author: Victor T. N.
"""


import numpy as np
import os
import trimesh
from human.utils.tfrecord import parse_record, decode_record
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
from star.tf.star import STAR  # noqa: E402
import tensorflow as tf  # noqa: E402


def view_recording(path_star="star",
                   poses=tf.zeros((1, 72), dtype=tf.float32),
                   betas=tf.zeros((1, 10), dtype=tf.float32),
                   trans=tf.zeros((1, 3), dtype=tf.float32),
                   dt=0.1, fps=10, gender="neutral",
                   face_color=(0, 255, 65, 255),
                   color_background=(255, 255, 255, 255),
                   resolution=(800, 600)):
    """Creates an animated visualization of a single AMASS recording.

    Args:
        path_star (str): Path to the directory containing the STAR models.
                         Defaults to "star".
        poses (tf.Tensor): Poses array from the recording.
                           Defaults to tf.zeros((1, 72)).
        betas (tf.Tensor): Betas (shape primitives).
                           Defaults to tf.zeros((1, 10)).
        trans (tf.Tensor): Translations array from the recording.
                           Must have the same length as "poses".
                           Defaults to tf.zeros((1, 3), dtype=tf.float32).
        dt (float): Time interval between frames. Defaults to 0.1.
        fps (int): Frames per second of the animation.
                   Rendering does not achieve high framerates.
                   Defaults to 10.
        gender (str): Gender of the recording. Defaults to "neutral".
        face_color (tuple): Face color of the human representation.
                            Defaults to green: (0, 255, 65, 255).
        color_background (tuple): Background color of the viewer.
                                  Defaults to white: (255, 255, 255, 255).
        resolution (tuple): Resolution of the viewer window.
                            Defaults to (800, 600).
    """
    # Check "gender" string to avoid problems with bytes-like representations
    if "female" in gender:
        gender = "female"
    elif "male" in gender:
        gender = "male"
    else:
        gender = "neutral"
    # Path to the STAR model
    path_model = os.path.join(path_star, str(gender) + ".npz")
    # Create STAR model
    star = STAR(path_model=path_model, num_betas=betas.shape[1])
    # Create iterator for the frame number
    frames = iter(range(0, poses.shape[0], int(1/(fps*dt))))
    fId = next(frames)
    # Create a body with the STAR model
    body = star(poses[fId:fId+1, :72], betas, trans[fId:fId+1])
    # Generate corresponding mesh
    body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                face_colors=face_color)
    # Create a scene with this mesh
    scene = trimesh.Scene(body_mesh)

    # Create a callback
    def update_scene(scene):
        fId = next(frames)
        # Create a body with the STAR model
        body = star(poses[fId:fId+1, :72], betas, trans[fId:fId+1])
        # Generate corresponding mesh
        body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                    face_colors=face_color)
        # Delete the previous mesh from the scene
        scene.delete_geometry("geometry_0")
        # Add the current geometry to the scene
        scene.add_geometry(body_mesh)

    # Create a viewer
    try:
        scene.show(callback=update_scene,
                   smooth=False,
                   background=color_background,
                   resolution=resolution,
                   callback_period=1/fps)
    except StopIteration:
        return


def view_test(path_star="star",
              poses_prediction=tf.zeros((1, 72), dtype=tf.float32),
              poses_ground_truth=tf.zeros((1, 72), dtype=tf.float32),
              betas=tf.zeros((1, 10), dtype=tf.float32),
              dt=0.1, fps=10, gender="neutral",
              color_prediction=(228, 175, 19, 200),
              color_ground_truth=(0, 255, 65, 200),
              color_background=(255, 255, 255, 255),
              resolution=(800, 600)):
    """Visualization function for testing, still in progress."""
    # Fixed translation tensors
    trans_prediction = tf.constant([[0, 0, 0]], dtype=tf.float32)
    trans_ground_truth = tf.constant([[2, 0, 0]], dtype=tf.float32)
    # Path to the STAR model
    path_model = os.path.join(path_star, str(gender) + ".npz")
    # Create STAR model
    star = STAR(path_model=path_model, num_betas=betas.shape[1])
    # Create iterator for the frame number
    frames = iter(range(0, poses_prediction.shape[0], int(1/(fps*dt))))
    fId = next(frames)
    # Create two bodies and corresponding meshes
    body_prediction = star(poses_prediction[fId:fId+1, :72],
                           betas, trans_prediction)
    mesh_prediction = trimesh.Trimesh(vertices=body_prediction[0],
                                      faces=star.f,
                                      face_colors=color_prediction)
    body_ground_truth = star(poses_ground_truth[fId:fId+1, :72],
                             betas, trans_ground_truth)
    mesh_ground_truth = trimesh.Trimesh(vertices=body_ground_truth[0],
                                        faces=star.f,
                                        face_colors=color_ground_truth)
    # Create a scene with this mesh
    scene = trimesh.Scene(mesh_prediction)
    scene.add_geometry(mesh_ground_truth)

    # Create a callback
    def update_scene(scene):
        fId = next(frames)
        # Create two bodies and corresponding meshes
        body_prediction = star(poses_prediction[fId:fId+1, :72],
                               betas, trans_prediction)
        mesh_prediction = trimesh.Trimesh(vertices=body_prediction[0],
                                          faces=star.f,
                                          face_colors=color_prediction)
        body_ground_truth = star(poses_ground_truth[fId:fId+1, :72],
                                 betas, trans_ground_truth)
        mesh_ground_truth = trimesh.Trimesh(vertices=body_ground_truth[0],
                                            faces=star.f,
                                            face_colors=color_ground_truth)
        # Delete the previous meshes from the scene
        scene.delete_geometry("geometry_0")
        scene.delete_geometry("geometry_1")
        # Add the current geometries to the scene
        scene.add_geometry(mesh_prediction)
        scene.add_geometry(mesh_ground_truth)

    # Create a viewer
    try:
        scene.show(callback=update_scene,
                   smooth=False,
                   background=color_background,
                   resolution=resolution,
                   callback_period=1/fps)
    except StopIteration:
        return


def view_amass_npz(path_npz="recording.npz", path_star="star"):
    """Visualize a single .npz file from AMASS.

    Args:
        path_npz (str): Full path to the .npz file to be visualized.
                        Defaults to "recording.npz".
        path_star (str): Path to the directory containing the STAR models.
                         Defaults to "star".
    """
    # Load the compressed NumPy file
    try:
        body_data = np.load(path_npz)
    except IOError:
        print("Error loading " + path_npz)
    except ValueError:
        print("allow_pickle=True required to load " + path_npz)
    else:
        view_recording(path_star=path_star,
                       poses=tf.constant(body_data["poses"][:, :72],
                                         dtype=tf.float32),
                       betas=tf.constant(body_data["betas"][:][np.newaxis],
                                         dtype=tf.float32),
                       trans=tf.constant(body_data["trans"][:],
                                         dtype=tf.float32),
                       dt=float(1/body_data["mocap_framerate"]),
                       gender=str(body_data["gender"]))


def view_tfrecord(path_tfrecord="dataset.tfrecord", path_star="star"):
    """Visualize a random recording from a TFRecord generated from AMASS.
    Notice that translation data is not contained in TFRecords, thus the
    virtual human stays fixed in the origin.

    Args:
        path_tfrecord (str): Full path to the TFRecord file to be visualized.
                             Defaults to "dataset.tfrecord".
        path_star (str): Path to the directory containing the STAR models.
                         Defaults to "star".
    """
    # Create a dataset
    dataset = tf.data.TFRecordDataset(path_tfrecord)
    # Shuffle the dataset
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    # Take a record as example
    record = next(iter(dataset))
    # Parse and decode
    poses, seq_length, betas, dt, gender = decode_record(parse_record(record))
    # View
    view_recording(path_star=path_star,
                   poses=poses, betas=betas,
                   trans=tf.zeros((poses.shape[0], 3), dtype=tf.float32),
                   dt=dt, gender=gender)
