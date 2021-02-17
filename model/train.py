"""
train.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the recommended AMASS splits for training, validation and testing.

Author: Victor T. N.
"""


import glob
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
from human import get_human_model  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import optimizers  # noqa: E402


def parse_record(record):
    """Parse a single record from the dataset.

    Args:
        record (raw dataset record): a single record from the dataset.

    Returns:
        (parsed record): a single parsed record from the dataset.
    """
    feature_names = {
        "poses": tf.io.VarLenFeature(tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "betas": tf.io.VarLenFeature(tf.float32),
        "gender": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


if __name__ == '__main__':
    # The HuMAn neural network
    model = get_human_model()
    # Create a decaying learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=1e5, decay_rate=0.96, staircase=True
    )
    # Compile the model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Data splits
    splits = ["train", "valid", "test"]
    # Create datasets
    dataset = {}
    for split in splits:
        # Full path to the datasets of a specific split, with wildcards
        tfr_paths = os.path.join(tfr_home, split, "*.tfrecord")
        # Expand with glob
        tfr_list = glob.glob(tfr_paths)
        # Load the TFRecords as a Dataset
        raw_ds = tf.data.TFRecordDataset(tfr_list)
        # Parse the recordings
        dataset[split] = raw_ds.map(parse_record)
        # Get a record as example
        record = next(iter(dataset[split]))
        print(f"Poses: {record['poses']}")
