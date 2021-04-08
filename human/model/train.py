"""train.py

Author: Victor T. N.
"""


import os
from datetime import datetime
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
import tensorflow_addons as tfa  # noqa: E402


SHUFFLE_BUFFER = 1000


def full_training_loop(model, train_datasets=[], valid_datasets=[],
                       seq_lengths=[], batch_sizes=[], swa=[], rlrp=[],
                       patience=[], name="noname", ckpt_dir="checkpoints",
                       log_dir="logs", save_dir="saves"):
    for i in range(len(train_datasets)):
        # Retrieve current date and time
        date = datetime.today().strftime("%Y-%m-%d-%H-%M")
        # Callbacks list
        callbacks = []
        if swa[i]:
            # Timestamp
            stamp = f"{date}_{name}_swa"
            # Optimizer (SGD + SWA)
            opt = tfa.optimizers.SWA(tf.keras.optimizers.SGD(
                    learning_rate=1e-5, momentum=0.5, nesterov=True))
            # Checkpoint callback
            callbacks.append(tfa.callbacks.AverageModelCheckpoint(
                update_weights=True, filepath=f"{ckpt_dir}/{stamp}",
                save_best_only=True, save_weights_only=True))
        else:
            # Timestamp
            stamp = f"{date}_{name}_{seq_lengths[i]}_{batch_sizes[i]}"
            # Optimizer (Adam)
            opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
            # Checkpoint callback
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{ckpt_dir}/{stamp}", save_best_only=True,
                save_weights_only=True))
        # Early stopping callback
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=patience[i]))
        # TensorBoard callback
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, stamp), profile_batch=0))
        # Reduce learning rate on plateau callback
        if rlrp[i]:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, patience=patience[i]))
        # Compile the model
        model.compile(loss="mse", metrics=["mae"], optimizer=opt)
        # Print useful information
        print(f"Sequence length: {seq_lengths[i]}\n"
              f"Batch size: {batch_sizes[i]}")
        if swa[i]:
            print("Optimizer: SGD + SWA")
        else:
            print("Optimizer: Adam")
        # Map and batch the dataset
        train_mapped = (train_datasets[i]
                        .map(dataset.map_train,
                             num_parallel_calls=tf.data.AUTOTUNE,
                             deterministic=False)
                        .shuffle(SHUFFLE_BUFFER)
                        .batch(batch_sizes[i])
                        .prefetch(tf.data.AUTOTUNE))
        valid_mapped = (valid_datasets[i]
                        .map(dataset.map_train,
                             num_parallel_calls=tf.data.AUTOTUNE,
                             deterministic=False)
                        .shuffle(SHUFFLE_BUFFER)
                        .batch(batch_sizes[i])
                        .prefetch(tf.data.AUTOTUNE))
        # Start training
        model.fit(x=train_mapped, epochs=20, callbacks=callbacks,
                  validation_data=valid_mapped)
    print(f"Training done for {name}. Saving model...")
    model.save_weights(os.path.join(save_dir, name))
