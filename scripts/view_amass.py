"""view_amass.py

This script shows how to use the visualization tools to generate an animation
directly from a .npz file from AMASS.

Author: Victor T. N.
"""

from human.utils.visualization import view_amass_npz


if __name__ == "__main__":
    path_star = "../../../AMASS/models/star"
    path_npz = ("../../../AMASS/datasets/Eyes_Japan_Dataset/"
                "hamada/accident-01-dodge-hamada_poses.npz")
    view_amass_npz(path_npz=path_npz, path_star=path_star)
