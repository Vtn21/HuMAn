"""
view_amass.py

This script provides an animated visualization for any AMASS recording.
Model gender is determined by the recording.
Modify the "npz_bdata_path" variable as required.
You can choose to lock global translations and rotations, showing only
    joint motions.
The pyglet renderer from trimesh seems to be a software renderer, thus
    it works at low framerates. It is still enough for basic visualization.

Author: Victor T. N.
"""

import numpy as np
import trimesh
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
from star.tf.star import STAR
import tensorflow as tf


FACE_COLORS = (0, 255, 65, 255)  # Color for mesh faces
BG_COLORS = (116, 120, 128, 255)  # Color for background
FPS = 10


def pbt_from_bdata(bdata, fId=0, lock_trans=False,
                   lock_orient=False, num_betas=0):
    # Body pose
    if lock_orient:
        pose = tf.constant(np.concatenate((np.zeros(1, 3),
                                           bdata['poses'][fId:fId+1, 3:72]),
                                          axis=1), dtype=tf.float32)
    else:
        pose = tf.constant(bdata['poses'][fId:fId+1, :72], dtype=tf.float32)
    # Betas (shape primitives)
    if num_betas <= 0 or num_betas > bdata['betas'].shape[0]:
        betas = tf.constant(bdata['betas'][:][np.newaxis], dtype=tf.float32)
    else:
        betas = tf.constant(bdata['betas'][:num_betas][np.newaxis],
                            dtype=tf.float32)
    # Body translation
    if lock_trans:
        trans = tf.constant(np.zeros(1, 3), dtype=tf.float32)
    else:
        trans = tf.constant(bdata['trans'][fId:fId+1], dtype=tf.float32)
    return pose, betas, trans


def update_scene(scene):
    fId = next(frames)
    # Load input information for the current frame
    pose, betas, trans = pbt_from_bdata(bdata, fId)
    # Generate a body with the STAR model
    body = star(pose, betas, trans)
    # Generate corresponding mesh
    body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                face_colors=FACE_COLORS)
    # Delete the previous mesh from the scene
    scene.delete_geometry("geometry_0")
    # Add the current mesh to the scene
    scene.add_geometry(body_mesh)


if __name__ == "__main__":

    # Path to the AMASS motion npz file
    # npz_bdata_path = "../../AMASS/datasets/KIT/3/912_3_01_poses.npz"
    npz_bdata_path = "../../AMASS/datasets/KIT/314/run04_poses.npz"
    # npz_bdata_path = "../../AMASS/datasets/KIT/314/parkour02_poses.npz"
    # npz_bdata_path = "../../AMASS/datasets/Eyes_Japan_Dataset/hamada/accident-01-dodge-hamada_poses.npz"
    # npz_bdata_path = "../../AMASS/datasets/BMLhandball/S03_Expert/Trial_upper_left_140_poses.npz"
    # npz_bdata_path = "../../AMASS/datasets/TCD_handMocap/ExperimentDatabase/OK_A_poses.npz"

    # Load the compressed numpy file
    bdata = np.load(npz_bdata_path)

    # Print info
    print('Data keys available:%s' % list(bdata.keys()))
    print('The subject of the mocap sequence is %s.' % bdata['gender'])
    print('Vector poses has %d elements for each of %d frames.' %
          (bdata['poses'].shape[1], bdata['poses'].shape[0]))
    print('Vector betas has %d elements constant for the whole sequence.' %
          bdata['betas'].shape[0])
    print('Vector trans has %d elements for each of %d frames.' %
          (bdata['trans'].shape[1], bdata['trans'].shape[0]))
    print('This mocap sequence has been recorded at %d Hz.' %
          (bdata['mocap_framerate']))

    path_model = os.path.join("../../AMASS/models/star",
                              "".join((str(bdata['gender']), ".npz")))

    # Create STAR model
    star = STAR(path_model=path_model, num_betas=bdata['betas'].shape[0])

    # Create an iterator for the frame number
    frames = iter(range(0, bdata['poses'].shape[0],
                        int(bdata['mocap_framerate']/FPS)))
    fId = next(frames)

    # Load input information for the current frame
    pose, betas, trans = pbt_from_bdata(bdata, fId)
    # Generate a body with the STAR model
    body = star(pose, betas, trans)
    # Generate corresponding mesh
    body_mesh = trimesh.Trimesh(vertices=body[0], faces=star.f,
                                face_colors=FACE_COLORS)
    # Create a scene with this mesh
    scene = trimesh.Scene(body_mesh)

    # Create a viewer
    scene.show(callback=update_scene,
               smooth=False,
               background=BG_COLORS,
               resolution=(800, 600),
               callback_period=1/FPS)
