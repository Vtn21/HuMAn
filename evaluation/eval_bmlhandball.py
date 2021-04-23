"""eval_bmlhandball.py

Author: Victor T. N.
"""


import glob
import numpy as np
import os
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


SAVES = ["train_universal", "train_bmlhandball", "transfer_bmlhandball"]


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Path where the saved models are located
    saves_home = "../training/saves"
    # The HuMAn neural network
    model = HuMAn()
    # Create groups of saved models and evaluation datasets
    saves = []
    datasets = []
    tags = []
    # Model: universal / Dataset: validation
    saves.append([os.path.join(saves_home, "train_universal")])
    datasets.append([dataset.folder_to_dataset(
        os.path.join(tfr_home, "valid_256"))])
    tags.append("universal_valid")
    # Model: universal / Dataset: BML
    saves.append([os.path.join(saves_home, "train_universal")])
    datasets.append([dataset.folder_to_dataset(
        os.path.join(tfr_home, "BMLhandball_256"))])
    tags.append("universal_bml")
    # Model: train BML / Dataset: validation
    saves.append([os.path.join(saves_home, f"train_bmlhandball_fold{fold}")
                  for fold in range(10)])
    datasets.append([dataset.folder_to_dataset(
        os.path.join(tfr_home, "valid_256")) for _ in range(10)])
    tags.append("train_valid")
    # Model: train BML / Dataset: BML
    saves.append([os.path.join(saves_home, f"train_bmlhandball_fold{fold}")
                  for fold in range(10)])
    datasets.append([dataset.tfrecords_to_dataset(record) for record in
                     glob.glob(os.path.join(tfr_home, "BMLhandball_256/*"))])
    tags.append("train_bml")
    # Model: transfer BML / Dataset: validation
    saves.append([os.path.join(saves_home, f"transfer_bmlhandball_fold{fold}")
                  for fold in range(10)])
    datasets.append([dataset.folder_to_dataset(
        os.path.join(tfr_home, "valid_256")) for _ in range(10)])
    tags.append("transfer_valid")
    # Model: transfer BML / Dataset: BML
    saves.append([os.path.join(saves_home, f"transfer_bmlhandball_fold{fold}")
                  for fold in range(10)])
    datasets.append([dataset.tfrecords_to_dataset(record) for record in
                     glob.glob(os.path.join(tfr_home, "BMLhandball_256/*"))])
    tags.append("transfer_bml")
    # Iterate through all groups
    for i in range(len(saves)):
        # Create variables to store mean and stdev for each group
        mean = 0.0
        stdev = 0.0
        # Create also a counter for the number of data points
        pts = 0
        # Iterate inside a single group
        for j in range(len(saves[i])):
            # Load model weights
            model.load_weights(saves[i][j])
            # Iterate through a number of horizon frames
            for horizon_frames in range(1, 11):
                # Load the evaluation dataset
                mapped_ds = datasets[i][j].map(lambda x: dataset.map_dataset(
                    x, skeleton="full_body", horizon_frames=horizon_frames),
                    num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                eval_ds = mapped_ds.batch(256).prefetch(tf.data.AUTOTUNE)
                # Predict
                print(f"Predicting for group {tags[i]} "
                      f"with horizon_frames={horizon_frames}")
                prediction = model.predict(eval_ds, verbose=1)
                # Compute the number of data points for a single recording
                rec_pts = prediction.shape[1]*prediction.shape[2]
                # Create the reference dataset
                # All sequences are joined in a single large batch
                reference_ds = (mapped_ds.batch(prediction.shape[0])
                                .prefetch(tf.data.AUTOTUNE))
                # Extract the values as NumPy arrays
                inputs, pose_targets = next(reference_ds.as_numpy_iterator())
                # Compute the absolute error between targets and predictions
                abs_err = np.abs(prediction - pose_targets)
                # Compute mean and standard deviation for each recording
                rec_mean = np.mean(abs_err, axis=(1, 2))
                rec_stdev = np.std(abs_err, axis=(1, 2))
                # Extract horizon times from "inputs"
                horizon_input = np.round(inputs["horizon_input"][:, 0, 0], 4)
                for n in range(horizon_input.shape[0]):
                    # A "horizon_input" zeroed out means an impossible shift
                    if horizon_input[n] != 0:
                        # Compute new values of mean and standard deviation
                        new_pts = pts + rec_pts
                        new_mean = (pts*mean + rec_pts*rec_mean[n]) / new_pts
                        new_stdev = np.sqrt((
                            pts*stdev**2 + rec_pts*rec_stdev[n]**2 +
                            pts*rec_pts*(mean - rec_mean[n])**2 / new_pts) /
                            new_pts)
                        # Update values
                        mean = new_mean
                        stdev = new_stdev
                        pts = new_pts
        # Save values for this group to a NumPy npz file
        np.savez(f"bmlhandball/{tags[i]}.npz", mean=mean, stdev=stdev)
