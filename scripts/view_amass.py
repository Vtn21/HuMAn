"""
view_amass.py

This script provides an animated visualization for any AMASS recording.
Model gender is determined by the recording.
Modify the "npz_path" variable as required.
You can choose to lock global translations and rotations, showing only
    joint motions.
The pyglet renderer from trimesh seems to be a software renderer, thus
    it works at low framerates. It is still enough for basic visualization.

Author: Victor T. N.
"""

from human.utils.visualization import view_amass_npz


if __name__ == "__main__":
    path_star = "../../../AMASS/models/star"
    path_npz = "../../../AMASS/datasets/Eyes_Japan_Dataset/hamada/accident-01-dodge-hamada_poses.npz"
    view_amass_npz(path_npz=path_npz, path_star=path_star)
