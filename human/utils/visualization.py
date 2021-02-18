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
                   background_color=(255, 255, 255, 255),
                   resolution=(800, 600)):
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
                   background=background_color,
                   resolution=resolution,
                   callback_period=1/fps)
    except StopIteration:
        return


def view_amass_npz(path_npz="recording.npz", path_star="star"):
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
                       gender=body_data["gender"])


def view_tfrecord(path_tfrecord="dataset.tfrecord", path_star="star"):
    # Create a dataset
    dataset = tf.data.TFRecordDataset(path_tfrecord)
    # Shuffle the dataset
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    # Take a record as example
    record = next(iter(dataset))
    # Parse and decode
    poses, betas, dt, gender = decode_record(parse_record(record))
    # View
    view_recording(path_star=path_star,
                   poses=poses, betas=betas,
                   trans=tf.zeros((poses.shape[0], 3), dtype=tf.float32),
                   dt=dt, gender=gender)
