"""view_tfrecord.py

This script shows how to use the visualization tools to generate an animation
of a random recording from a TFRecord file generated from the AMASS dataset.

This is important for showing if the data loaded into the TFRecord files are
intact and correctly represent the human motion recordings.

Author: Victor T. N.
"""


from human.utils.visualization import view_tfrecord


if __name__ == "__main__":
    path_star = "../../../AMASS/models/star"
    path_tfrecord = ("../../../AMASS/tfrecords/train/"
                     "Eyes_Japan_Dataset.tfrecord")
    view_tfrecord(path_tfrecord=path_tfrecord, path_star=path_star)
