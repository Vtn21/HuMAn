"""
view_tfrecord.py

Visualize the AMASS TFRecords created with "amass_to_tfrecord.py".
This script is important for showing if the data loaded into the
    TFRecord files still represent the human motion recordings.

Author: Victor T. N.
"""


from human.utils.visualization import view_tfrecord


if __name__ == "__main__":
    path_star = "../../../AMASS/models/star"
    path_tfrecord = "../../../AMASS/tfrecords/train/Eyes_Japan_Dataset.tfrecord"
    view_tfrecord(path_tfrecord=path_tfrecord, path_star=path_star)
