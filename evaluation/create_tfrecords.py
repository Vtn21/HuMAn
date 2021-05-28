"""create_tfrecords.py

Creates TFRecords files, with the same structure as the ones used for training
and validation, but containing a selected recording from AMASS. They are used
to generate time plots.

Author: Victor T. N.
"""


import os
import sys
from human.utils.preprocessing import amass_to_tfrecord_simple


if __name__ == '__main__':
    # Path to the datasets
    amass_home = "../../AMASS/datasets"
    # Path to save the TFRecords files
    tfr_home = "../../AMASS/tfrecords/time_plots"
    # Create the directory, if it doesn't exist
    try:
        os.makedirs(tfr_home)
    except FileExistsError:
        print(f"Directory {tfr_home} already exists and will be used.")
    except FileNotFoundError:
        print(f"Directory {tfr_home} is invalid.")
        sys.exit()
    else:
        print(f"Creating new directory {tfr_home}.")
    # Input npz files to read
    input_npzs = ["HumanEva/S1/Walking_3_poses.npz",
                  "BMLHandball/S01_Expert/Trial_upper_left_right_176_poses.npz",  # noqa: E501
                  "BMLHandball/S02_Novice/Trial_upper_left_right_176_poses.npz",  # noqa: E501
                  "BMLHandball/S04_Expert/Trial_upper_left_right_176_poses.npz",  # noqa: E501
                  "BMLHandball/S06_Novice/Trial_upper_left_right_176_poses.npz",  # noqa: E501
                  "BMLHandball/S07_Expert/Trial_upper_left_right_176_poses.npz",  # noqa: E501
                  "MPI_HDM05/bk/HDM_bk_01-04_03_120_poses.npz",
                  "MPI_HDM05/dg/HDM_dg_01-04_03_120_poses.npz",
                  "MPI_HDM05/mm/HDM_mm_01-04_03_120_poses.npz",
                  "MPI_HDM05/tr/HDM_tr_01-04_01_120_poses.npz",
                  "Eyes_Japan_Dataset/kawaguchi/accident-02-dodge fast-kawaguchi_poses.npz"]  # noqa: E501
    # Name of the output TFRecords (one for each npz file)
    output_tfrecords = ["HumanEva_Walking",
                        "BMLHandball_S01_Expert",
                        "BMLHandball_S02_Novice",
                        "BMLHandball_S04_Expert",
                        "BMLHandball_S06_Novice",
                        "BMLHandball_S07_Expert",
                        "MPI_HDM05_bk",
                        "MPI_HDM05_dg",
                        "MPI_HDM05_mm",
                        "MPI_HDM05_tr",
                        "Eyes_Japan_dodge"]
    for input_npz, output_tfrecord in zip(input_npzs, output_tfrecords):
        input_npz = os.path.join(amass_home, input_npz)
        output_tfrecord = os.path.join(tfr_home, output_tfrecord + ".tfrecord")
        amass_to_tfrecord_simple(input_npz, output_tfrecord)
