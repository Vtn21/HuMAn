"""
train_universal.py

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


SHUFFLE_BUFFER = 1000


if __name__ == "__main__":
    # File paths
    tfr_home = "../../AMASS/tfrecords"
    ckpt_path = "checkpoints/train_universal"
    log_path = "logs"
    # Sequence of hyperparameters
    seq_len =  [256,  256,    256,   512,   512,  1024,   256]  # noqa: E222
    batch_size = [8,   16,     32,     8,    16,     8,    32]
    swa =    [False, False, False, False, False, False,  True]  # noqa: E222
    rlrp =   [False, False,  True, False,  True,  True, False]  # noqa: E222
    # Build a dataset for adapting the normalization layer
    norm_folder = os.path.join(tfr_home, "train_256")
    norm_ds = (dataset.folder_to_dataset(norm_folder)
               .map(dataset.map_pose_input,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))
    # Instantiate the HuMAn neural network
    model = HuMAn(norm_dataset=norm_ds)
    # Automatically restore weights
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    # Parse the latest checkpoint and define where training will start
    start = 0
    if latest_ckpt is not None:
        latest_ckpt_split = latest_ckpt.split("_")
        if latest_ckpt_split[-1] == "swa":
            start = len(seq_len) - 1
        else:
            for _ in range(len(seq_len)):
                if seq_len[start] == latest_ckpt_split[-2]:
                    if batch_size[start] == latest_ckpt_split[-1]:
                        break
                else:
                    start += 1
        # Restore weights
        print("Restoring model from " + latest_ckpt)
        model.load_weights(latest_ckpt)
    # Compile the model with the Adam optimizer
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss="mse", metrics=["mae"], optimizer=opt)
    # Iterate through the hyperparameters sequence and train
    for i in range(start, len(seq_len)):
        mapped_ds = {}
        for split in ["train", "valid"]:
            # Load the dataset
            ds_folder = os.path.join(tfr_home, f"{split}_{seq_len[i]}")
            mapped_ds[split] = (dataset.folder_to_dataset(ds_folder)
                                .map(dataset.map_train,
                                     num_parallel_calls=tf.data.AUTOTUNE,
                                     deterministic=False)
                                .shuffle(SHUFFLE_BUFFER)
                                .batch(batch_size[i])
                                .prefetch(tf.data.AUTOTUNE))
        # Retrieve current date and time
        date = datetime.today().strftime("%Y-%m-%d-%H-%M")
        if swa[i]:
            stamp = f"{date}_train_universal_swa"
            # SWA optimizer
            opt = tfa.optimizers.SWA(optimizers.SGD(
                    learning_rate=1e-5, momentum=0.5, nesterov=True),
                    average_period=500)
            # Recompile the model to replace Adam with SWA
            model.compile(loss="mse", metrics=["mae"], optimizer=opt)
            # Checkpoint callback
            checkpoint = tfa.callbacks.AverageModelCheckpoint(
                update_weights=True, filepath=f"{ckpt_path}/{stamp}",
                save_best_only=True, save_weights_only=True)
        else:
            stamp = f"{date}_train_universal_{seq_len[i]}_{batch_size[i]}"
            # Checkpoint callback
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{ckpt_path}/{stamp}", save_best_only=True,
                save_weights_only=True)
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(patience=1)
        # TensorBoard
        log_dir = f"logs/{stamp}"
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     profile_batch=0)
        # Reduce learning rate on plateau
        if rlrp[i]:
            rlrp = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,
                                                        patience=0)
            callbacks = [checkpoint, early_stop, tensorboard, rlrp]
        else:
            callbacks = [checkpoint, early_stop, tensorboard]
        # Train the model
        if swa[i]:
            print("Training the model with Stochastic Weight Averaging (SWA)")
        else:
            print(f"Training model with sequence length {seq_len[i]} "
                  f"and batch size {batch_size[i]}")
        model.fit(x=mapped_ds["train"], epochs=20, callbacks=callbacks,
                  validation_data=mapped_ds["valid"])
    print("Done.")
