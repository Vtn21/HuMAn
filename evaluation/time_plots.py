"""time_plots.py

Uses the TFRecords created with "create_tfrecords.py" to generate predictions
and store them as time series (in .npz files).

Author: Victor T. N.
"""


import numpy as np
import os
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


FRAMERATE = 120  # Hz
HORIZON = 0.1  # second
CASES = [("HumanEva_Walking",       "train_universal",            "full_body"),
         ("HumanEva_Walking",       "train_universal",            "legs_arms"),
         ("HumanEva_Walking",       "train_universal",            "legs"),
         ("BMLHandball_S01_Expert", "train_universal",            "full_body"),
         ("BMLHandball_S01_Expert", "train_bmlhandball_fold0",    "full_body"),
         ("BMLHandball_S01_Expert", "transfer_bmlhandball_fold0", "full_body"),
         ("BMLHandball_S02_Novice", "train_universal",            "full_body"),
         ("BMLHandball_S02_Novice", "train_bmlhandball_fold1",    "full_body"),
         ("BMLHandball_S02_Novice", "transfer_bmlhandball_fold1", "full_body"),
         ("BMLHandball_S04_Expert", "train_universal",            "full_body"),
         ("BMLHandball_S04_Expert", "train_bmlhandball_fold3",    "full_body"),
         ("BMLHandball_S04_Expert", "transfer_bmlhandball_fold3", "full_body"),
         ("BMLHandball_S06_Novice", "train_universal",            "full_body"),
         ("BMLHandball_S06_Novice", "train_bmlhandball_fold5",    "full_body"),
         ("BMLHandball_S06_Novice", "transfer_bmlhandball_fold5", "full_body"),
         ("BMLHandball_S07_Expert", "train_universal",            "full_body"),
         ("BMLHandball_S07_Expert", "train_bmlhandball_fold6",    "full_body"),
         ("BMLHandball_S07_Expert", "transfer_bmlhandball_fold6", "full_body"),
         ("MPI_HDM05_bk",           "train_mpihdm05_bk",          "full_body"),
         ("MPI_HDM05_dg",           "train_mpihdm05_dg",          "full_body"),
         ("MPI_HDM05_mm",           "train_mpihdm05_mm",          "full_body"),
         ("MPI_HDM05_tr",           "train_mpihdm05_tr",          "full_body"),
         ("Eyes_Japan_dodge",       "train_universal",            "full_body")]


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords/time_plots"
    # Path where the models are saved
    saves_path = "../training/saves"
    # Load the HuMAn neural network
    # Expects a normalization layer already adapted
    model = HuMAn()
    # Compute the number of frames for the selected horizon
    horizon_frames = int(HORIZON * FRAMERATE)
    # Create a counter for the cases
    counter = 0
    # Iterate through all cases
    for record, save, skeleton in CASES:
        # Build the model path
        save_path = os.path.join(saves_path, save)
        # Load weights from saved model
        model.load_weights(save_path)
        # Build the TFRecord path
        tfr_path = os.path.join(tfr_home, record + ".tfrecord")
        # Load the dataset
        parsed_ds = dataset.tfrecords_to_dataset([tfr_path])
        mapped_ds = parsed_ds.map(lambda x: dataset.map_dataset(
            x, skeleton=skeleton, horizon_frames=horizon_frames),
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        eval_ds = mapped_ds.batch(1).prefetch(tf.data.AUTOTUNE)
        # Retrieve the input
        inputs, pose_targets = next(eval_ds.as_numpy_iterator())
        # Predict
        print(f"Case {counter}: recording={record}, model={save}, "
              f"skeleton={skeleton}")
        predictions = model.predict(eval_ds, verbose=1)
        # Save data to a .npz file
        np.savez(f"time_plots/case{counter}.npz",
                 recording=record, model=save, skeleton=skeleton,
                 inputs=np.squeeze(inputs["pose_input"]),
                 targets=np.squeeze(pose_targets),
                 predictions=np.squeeze(predictions))
        # Increment counter
        counter += 1
