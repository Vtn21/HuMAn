"""eval_joints.py

Evaluates the universal model with the validation dataset, averaging absolute
errors over each joint (set of three angles).

Results are stored in npz files inside the "joints" subfolder.

Author: Victor T. N.
"""

import numpy as np
import os
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


# This dict configures the number of future frames for each sampling time
SAMPLING_FRAMES = {0.0833: range(1,   7, 1),
                   0.0667: range(1,   8, 1),
                   0.05:   range(1,  11, 1),
                   0.0417: range(1,  12, 2),
                   0.04:   range(1,  13, 2),
                   0.0333: range(1,  16, 2),
                   0.02:   range(1,  26, 4),
                   0.0167: range(1,  30, 4),
                   0.016:  range(1,  32, 4),
                   0.01:   range(1,  51, 5),
                   0.0083: range(1,  61, 5),
                   0.008:  range(1,  63, 5),
                   0.004:  range(1, 126, 10)}
# Create lists to store all possible combinations
HORIZON_FRAMES = []
# Fill the lists
for key, value in SAMPLING_FRAMES.items():
    HORIZON_FRAMES.extend(list(value))
# Turn lists into sets, removing repeated values, then turn into NumPy arrays
HORIZON_FRAMES = np.array(list(set(HORIZON_FRAMES)))

# Name of all joints
JOINTS = ["Root", "Left hip", "Right hip", "Low spine", "Left knee",
          "Right knee", "Mid spine", "Left ankle", "Right ankle",
          "High spine", "Left foot", "Right foot", "Neck", "Left clavicle",
          "Right clavicle", "Head", "Left shoulder", "Right shoulder",
          "Left elbow", "Right elbow", "Left wrist", "Right wrist",
          "Left hand", "Right hand"]


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the validation dataset
    parsed_ds = dataset.folder_to_dataset(
        os.path.join(tfr_home, "valid_256"))
    # Load the HuMAn neural network
    # Expects a normalization layer already adapted
    model = HuMAn()
    # Load weights from saved model
    saves_path = "../training/saves/train_universal"
    model.load_weights(saves_path)
    # Create NumPy arrays to store mean and stdev for each joint
    mean = np.zeros(len(JOINTS))
    stdev = np.zeros(len(JOINTS))
    # Create another array, to accumulate the number of data points
    pts = np.zeros(len(JOINTS))
    # Iterate through all specified horizon frames
    for horizon_frames in HORIZON_FRAMES:
        # Load validation data
        mapped_ds = parsed_ds.map(lambda x: dataset.map_dataset(
            x, skeleton="full_body", horizon_frames=horizon_frames),
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        # Create a dataset for evaluation
        eval_ds = mapped_ds.batch(256).prefetch(tf.data.AUTOTUNE)
        # Predict for the whole dataset
        print(f"Predicting with horizon_frames={horizon_frames}")
        prediction = model.predict(eval_ds, verbose=1)
        # Create a dataset to be used as reference
        # All sequences are joined in a single large batch
        reference_ds = (mapped_ds.batch(prediction.shape[0])
                        .prefetch(tf.data.AUTOTUNE))
        # Extract the values as NumPy arrays
        inputs, pose_targets = next(reference_ds.as_numpy_iterator())
        # Compute the absolute error between targets and predictions
        abs_err = np.abs(prediction - pose_targets)
        # Extract horizon times from "inputs"
        horizon_input = np.round(inputs["horizon_input"][:, 0, 0], 4)
        for i in range(horizon_input.shape[0]):
            # A "horizon_input" zeroed out means an impossible shift
            if horizon_input[i] != 0:
                for j in range(len(JOINTS)):
                    # Compute mean and stdev for a single joint
                    joint_mean = np.mean(abs_err[3*j:3*j+3])
                    joint_stdev = np.std(abs_err[3*j:3*j+3])
                    joint_pts = 3*256
                    # Compute new values
                    new_pts = pts[j] + joint_pts
                    new_mean = (pts[j]*mean[j] +
                                joint_pts*joint_mean) / new_pts
                    new_stdev = np.sqrt((
                        pts[j]*stdev[j]**2 + joint_pts*joint_stdev**2 +
                        pts[j]*joint_pts*(mean[j] - joint_mean)**2 /
                        new_pts) / new_pts)
                    # Update values
                    mean[j] = new_mean
                    stdev[j] = new_stdev
                    pts[j] = new_pts
    # Save results
    np.savez("joints/joints.npz", joints=JOINTS, mean=mean, stdev=stdev)
