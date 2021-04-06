"""train_bmlhandball.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the BMLhandball dataset, for learning task-specific motions.

Procedure types (control using the "PROCEDURE" global variable):
- Train: train the model from scratch.
- Transfer: fine-tune the universal model created using "train_universal.py".

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
PROCEDURE = "train"
# PROCEDURE = "transfer"


if __name__ == "__main__":
    # File paths
    tfr_home = "../../AMASS/tfrecords"
    ckpt_path = f"checkpoints/{PROCEDURE}_bmlhandball"
    log_path = "logs"
    # Training and validation subjects
    subjects = {"train": ["S01_Expert", "S02_Novice", "S04_Expert",
                          "S06_Novice", "S07_Expert", "S08_Novice",
                          "S09_Novice", "S10_Expert"],
                "valid": ["S03_Expert", "S05_Novice"]}
    # Sequence of hyperparameters
    seq_len =  [256,  256,    256,   512,   512,   256]  # noqa: E222
    batch_size = [8,   16,     32,     8,    16,    32]
    swa =    [False, False, False, False, False,  True]  # noqa: E222
    rlrp =   [False, False,  True, False,  True, False]  # noqa: E222
    # Instantiate the HuMAn neural network
    model = HuMAn()
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
    elif PROCEDURE == "transfer":
        universal_path = "checkpoints/train_universal"
        latest_universal = tf.train.latest_checkpoint(universal_path)
        # Restore weights
        print("Loading universal model from " + latest_universal)
        model.load_weights(latest_universal)
    # Compile the model with the Adam optimizer
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss="mse", metrics=["mae"], optimizer=opt)
    # Iterate through the hyperparameters sequence and train
    for i in range(start, len(seq_len)):
        mapped_ds = {}
        for split in ["train", "valid"]:
            tfr_list = []
            for subject in subjects[split]:
                tfr_list.append(os.path.join(tfr_home,
                                             f"BMLhandball_{seq_len[i]}",
                                             f"{subject}.tfrecord"))
            mapped_ds[split] = (dataset.tfrecords_to_dataset(tfr_list=tfr_list)
                                .map(dataset.map_train,
                                     num_parallel_calls=tf.data.AUTOTUNE,
                                     deterministic=False)
                                .shuffle(SHUFFLE_BUFFER)
                                .batch(batch_size[i])
                                .prefetch(tf.data.AUTOTUNE))
        # Retrieve current date and time
        date = datetime.today().strftime("%Y-%m-%d-%H-%M")
        if swa[i]:
            stamp = f"{date}_{PROCEDURE}_bmlhandball_swa"
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
            stamp = (f"{date}_{PROCEDURE}_bmlhandball_{seq_len[i]} "
                     f"_{batch_size[i]}")
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
