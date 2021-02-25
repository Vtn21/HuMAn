"""
train.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the recommended AMASS splits for training, validation and testing.

Author: Victor T. N.
"""


import os
import time
from human.model.human import get_human_model
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow.keras import optimizers  # noqa: E402


os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 32


if __name__ == '__main__':
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the TFRecords into datasets
    parsed_ds = dataset.load_all_splits(tfr_home)
    # Create mapped datasets
    mapped_ds = {}
    for split, ds in parsed_ds.items():
        mapped_ds[split] = (ds
                            .map(dataset.map_dataset,
                                 num_parallel_calls=tf.data.AUTOTUNE,
                                 deterministic=False)
                            .shuffle(SHUFFLE_BUFFER)
                            .batch(BATCH_SIZE)
                            .prefetch(tf.data.AUTOTUNE))
    # Load only the training pose inputs, to adapt the Normalization layer
    norm_ds = (parsed_ds["train"]
               .map(dataset.map_pose_input,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))
    # The HuMAn neural network
    model = get_human_model(norm_ds)
    # Create a decaying learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        5e-3, decay_steps=1e5, decay_rate=0.96, staircase=True)
    # Compile the model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Create a checkpoint callback
    ckpt_path = "checkpoints/ckpt_{epoch}"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path)
    # Create a TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{int(time.time())}", update_freq=100,
        profile_batch=(100, 500))
    # Create an early stopping callback (based on validation loss)
    early_stop = tf.keras.callbacks.EarlyStopping()
    # Train the model
    model.fit(x=mapped_ds["train"], epochs=10,
              callbacks=[checkpoint, tensorboard, early_stop],
              validation_data=mapped_ds["valid"])
