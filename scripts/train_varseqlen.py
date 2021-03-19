"""
train_varseqlen.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the recommended AMASS splits for training, validation and testing.
This script uses varied sequence lengths during training.

Author: Victor T. N.
"""


import os
import re
import time
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow.keras import optimizers  # noqa: E402


os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
SHUFFLE_BUFFER = 1000
SEQ_LENGTH = [1024, 512, 256]
BATCH_SIZE = [16, 32, 64]


if __name__ == '__main__':
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Create a list of split names
    train_splits = []
    valid_splits = []
    for seq_len in SEQ_LENGTH:
        train_splits.append(f"train_{seq_len}")
        valid_splits.append(f"valid_{seq_len}")
    splits = train_splits + valid_splits
    # Load the TFRecords into datasets
    parsed_train = dataset.load_splits(tfr_home, splits=train_splits)
    parsed_valid = dataset.load_splits(tfr_home, splits=valid_splits)
    # Create mapped datasets
    mapped_ds = {}
    for parsed_ds in [parsed_train, parsed_valid]:
        for i, (split, ds) in enumerate(parsed_ds.items()):
            mapped_ds[split] = (ds
                                .map(dataset.map_train,
                                     num_parallel_calls=tf.data.AUTOTUNE,
                                     deterministic=False)
                                .shuffle(SHUFFLE_BUFFER)
                                .batch(BATCH_SIZE[i])
                                .prefetch(tf.data.AUTOTUNE))
    # Load only the training pose inputs, to adapt the Normalization layer
    norm_ds = (parsed_train[train_splits[0]]
               .map(dataset.map_pose_input,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .batch(BATCH_SIZE[0])
               .prefetch(tf.data.AUTOTUNE))
    # The HuMAn neural network
    model = HuMAn(norm_dataset=norm_ds)
    # Load weights from checkpoints, if appropriate
    ckpt_path = "checkpoints"
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    restore = 0
    if latest_ckpt is not None:
        print("Restoring model from " + latest_ckpt)
        restore = int(re.sub("[^0-9]", "", latest_ckpt))
        model.load_weights(latest_ckpt)
    # Create a decaying learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=5e3, decay_rate=0.95, staircase=True)
    # Compile the model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Create a checkpoint callback
    ckpt_name = ckpt_path + f"/ckpt_{restore}" + "_{epoch}"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_name,
                                                    save_weights_only=True,
                                                    save_best_only=True)
    # Create a TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{int(time.time())}", update_freq=100,
        profile_batch=0)
    # Create an early stopping callback (based on validation loss)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=2)
    # Train the model
    i = 2
    print(f"Using {SEQ_LENGTH[i]} frames sequence length")
    model.fit(x=mapped_ds[train_splits[i]], epochs=5,
              callbacks=[checkpoint, tensorboard, early_stop],
              validation_data=mapped_ds[valid_splits[i]])
