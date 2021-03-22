"""
train_general.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the "train" and "valid" splits, adapted from the recommended AMASS
splits.

Author: Victor T. N.
"""


import os
from datetime import datetime
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
import tensorflow_addons as tfa  # noqa: E402
from tensorflow.keras import optimizers  # noqa: E402


LENGTH_BATCH = {256: [4, 8, 16, 32],
                512: [4, 8, 16],
                1024: [4, 8]}
SHUFFLE_BUFFER = 1000


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Build strings referencing each split
    splits = []
    for split in ["train", "valid"]:
        for seq_len in LENGTH_BATCH.keys():
            splits.append(f"{split}_{seq_len}")
    # Load TFRecords into datasets
    parsed_ds = dataset.load_splits(tfr_home, splits=splits)
    # Build a dataset for adapting the normalization layer
    norm_ds = (parsed_ds["train_256"]
               .map(dataset.map_pose_input,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .batch(LENGTH_BATCH[256][-1])
               .prefetch(tf.data.AUTOTUNE))
    # Instantiate the HuMAn neural network
    model = HuMAn(norm_dataset=norm_ds)
    # Compile the model
    model.compile(loss="mse", metrics=["mae"],
                  optimizer=optimizers.Adam(learning_rate=1e-3))
    # Iterate through the sequence lengths
    for seq_len in LENGTH_BATCH.keys():
        for i, batch_size in enumerate(LENGTH_BATCH[seq_len]):
            mapped_ds = {}
            for split in ["train", "valid"]:
                mapped_ds[split] = (parsed_ds[f"{split}_{seq_len}"]
                                    .map(dataset.map_train,
                                         num_parallel_calls=tf.data.AUTOTUNE,
                                         deterministic=False)
                                    .shuffle(SHUFFLE_BUFFER)
                                    .batch(batch_size)
                                    .prefetch(tf.data.AUTOTUNE))
            # Create callbacks
            date = datetime.today().strftime("%Y-%m-%d-%H-%M")
            stamp = f"{date}_train_{seq_len}_{batch_size}"
            # Checkpoint
            filepath = f"checkpoints/{stamp}"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath, save_best_only=True, save_weights_only=True)
            # Early stopping
            early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
            # TensorBoard
            log_dir = f"logs/{stamp}"
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq=100, profile_batch=0)
            # Reduce learning rate on plateau
            rlrp = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=1)
            # Select callbacks
            callbacks = [checkpoint, early_stop, tensorboard]
            if i == len(LENGTH_BATCH[seq_len]):
                # The last batch size allows learning rate decay
                callbacks.append(rlrp)
            # Train the model
            print(f"Training model with sequence length {seq_len} "
                  f"and batch size {batch_size}")
            model.fit(x=mapped_ds["train"], epochs=20, callbacks=callbacks,
                      validation_data=mapped_ds["valid"])
            # Clear session, to allow training on the next loop
            print("Clearing session...")
            tf.keras.backend.clear_session()
    # Run a final training, with Stochastic Weight Averaging (SWA)
    swa_ds = {}
    seq_len = 256
    for split in ["train", "valid"]:
        swa_ds[split] = (parsed_ds[f"{split}_{seq_len}"]
                         .map(dataset.map_train,
                              num_parallel_calls=tf.data.AUTOTUNE,
                              deterministic=False)
                         .shuffle(SHUFFLE_BUFFER)
                         .batch(LENGTH_BATCH[seq_len][-1])
                         .prefetch(tf.data.AUTOTUNE))
    # Recompile the model
    model.compile(loss="mse", metrics=["mae"],
                  optimizer=tfa.optimizers.SWA(optimizers.SGD(
                      learning_rate=1e-4, momentum=0.5, nesterov=True)))
    # Create callbacks
    date = datetime.today().strftime("%Y-%m-%d-%H-%M")
    stamp = f"{date}_train_swa"
    # Checkpoint
    filepath = f"checkpoints/{stamp}"
    checkpoint = tfa.callbacks.AverageModelCheckpoint(
        update_weights=True, filepath=filepath, save_best_only=True,
        save_weights_only=True)
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
    # TensorBoard
    log_dir = f"logs/{stamp}"
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq=100, profile_batch=0)
    # Reduce learning rate on plateau
    rlrp = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=1)
    # Select callbacks
    callbacks = [checkpoint, early_stop, tensorboard, rlrp]
    # Train the model
    print("Training the model with Stochastic Weight Averaging (SWA)")
    model.fit(x=mapped_ds["train"], epochs=20, callbacks=callbacks,
              validation_data=mapped_ds["valid"])
    # Clear session, to allow training on the next loop
    print("Clearing session...")
    tf.keras.backend.clear_session()
    # Save the final model
    date = datetime.today().strftime("%Y-%m-%d-%H-%M")
    stamp = f"{date}_train"
    print("Saving model...")
    model.save_weights(f"saves/{stamp}")
    print("Done.")
