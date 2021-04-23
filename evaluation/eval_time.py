"""eval_time.py

Evaluates the universal model using the 1024 frames validation dataset, to
check how error evolves over time.

Results are stored in npz files inside the "time" subfolder.

Author: Victor T. N.
"""


import numpy as np
import os
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the validation dataset
    parsed_ds = dataset.folder_to_dataset(
        os.path.join(tfr_home, "valid_1024"))
    # Load the HuMAn neural network
    model = HuMAn()
    # Load weights from saved model
    saves_path = "../training/saves/train_universal"
    model.load_weights(saves_path)
    # Create variables to store mean and stdev for each time frame'
    mean = np.zeros(1024)
    stdev = np.zeros(1024)
    # Create another array, to accumulate the number of data points
    pts = np.zeros(1024)
    # Iterate through a number of horizon frames
    for horizon_frames in range(1, 11):
        # Load the evaluation dataset
        mapped_ds = parsed_ds.map(lambda x: dataset.map_dataset(
            x, skeleton="full_body", horizon_frames=horizon_frames),
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        eval_ds = mapped_ds.batch(64).prefetch(tf.data.AUTOTUNE)
        # Predict
        print(f"Predicting with horizon_frames={horizon_frames}")
        prediction = model.predict(eval_ds, verbose=1)
        # Create the reference dataset
        # All sequences are joined in a single large batch
        reference_ds = (mapped_ds.batch(prediction.shape[0])
                        .prefetch(tf.data.AUTOTUNE))
        # Extract the values as NumPy arrays
        inputs, pose_targets = next(reference_ds.as_numpy_iterator())
        # Compute the absolute error between targets and predictions
        abs_err = np.abs(prediction - pose_targets)
        # Extract horizon time from "inputs"
        horizon_input = np.round(inputs["horizon_input"][:, 0, 0], 4)
        # Look for invalid predictions
        delete = []
        for n in range(horizon_input.shape[0]):
            # A "horizon_input" zeroed out means an impossible shift
            if horizon_input[n] == 0:
                # Mark for deletion
                delete.append(n)
        # Remove the invalid predicitons
        abs_err = np.delete(abs_err, delete, axis=0)
        # Compute the number of data points for a single time step
        step_pts = abs_err.shape[0]*abs_err.shape[2]
        # Compute mean and standard deviation for each time step
        step_mean = np.mean(abs_err, axis=(0, 2))
        step_stdev = np.std(abs_err, axis=(0, 2))
        # Compute new global values
        new_pts = pts + step_pts
        new_mean = (pts*mean + step_pts*step_mean) / new_pts
        new_stdev = np.sqrt((
            pts*stdev**2 + step_pts*step_stdev**2 +
            pts*step_pts*(mean - step_mean)**2 / new_pts) /
            new_pts)
        # Update
        mean = new_mean
        stdev = new_stdev
        pts = new_pts
    # Save values to a NumPy npz file
    np.savez("time/time.npz", mean=mean, stdev=stdev)
