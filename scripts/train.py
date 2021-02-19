"""
train.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the recommended AMASS splits for training, validation and testing.

Author: Victor T. N.
"""


import os
from human.model.human import get_human_model
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow.keras import optimizers  # noqa: E402


if __name__ == '__main__':
    # The HuMAn neural network
    model = get_human_model()
    # Create a decaying learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=1e5, decay_rate=0.96, staircase=True)
    # Compile the model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the TFRecords into datasets
    parsed_ds = dataset.load_all_splits(tfr_home)
    # Create mapped datasets
    mapped_ds = {}
    for split, ds in parsed_ds.items():
        mapped_ds[split] = ds.map(dataset.map_dataset)
        # Prepare the dataset for usage
        # Note: variable length recordings disallow batching
        mapped_ds[split] = (mapped_ds[split]
                            .shuffle(1000, reshuffle_each_iteration=True)
                            .prefetch(-1))
    # Test
    for input_example, target_example in mapped_ds["test"].take(1):
        example_predictions = model(input_example)
        print(example_predictions.shape)
