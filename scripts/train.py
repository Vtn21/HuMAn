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


SHUFFLE_BUFFER = 1000


if __name__ == '__main__':
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the TFRecords into datasets
    parsed_ds = dataset.load_all_splits(tfr_home)
    # Create mapped datasets
    mapped_ds = {}
    for split, ds in parsed_ds.items():
        mapped_ds[split] = (ds.map(dataset.map_dataset)
                            .shuffle(SHUFFLE_BUFFER)
                            .batch(1).prefetch(-1))
    # Load only the training pose inputs, to adapt the Normalization layer
    normalization_ds = (parsed_ds["train"].map(dataset.map_pose_input)
                        .batch(1).prefetch(-1))
    # The HuMAn neural network
    model = get_human_model(normalization_ds)
    # Create a decaying learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=1e5, decay_rate=0.96, staircase=True)
    # Compile the model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Create a checkpoint callback
    ckpt_path = "checkpoints/ckpt_{epoch}"
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path)
    # Create a TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")
    # Train the model
    model.fit(x=mapped_ds["train"], epochs=10,
              callbacks=[ckpt_cb, tensorboard],
              validation_data=mapped_ds["valid"])
