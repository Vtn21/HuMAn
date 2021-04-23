"""eval_universal.py



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
                   0.0417: range(1,  12, 1),
                   0.04:   range(1,  13, 1),
                   0.0333: range(1,  16, 1),
                   0.02:   range(1,  26, 2),
                   0.0167: range(1,  30, 2),
                   0.016:  range(1,  32, 2),
                   0.01:   range(1,  51, 3),
                   0.0083: range(1,  61, 3),
                   0.008:  range(1,  63, 3),
                   0.004:  range(1, 126, 6)}
# Create lists to store all possible combinations
SAMPLING_TIME = []
HORIZON_FRAMES = []
HORIZON_TIME = []
# Fill the lists
for key, value in SAMPLING_FRAMES.items():
    SAMPLING_TIME.extend([key for _ in value])
    HORIZON_FRAMES.extend(list(value))
    HORIZON_TIME.extend([round(val*key, 4) for val in value])
SAMPLING_TIME = np.array(SAMPLING_TIME, dtype=np.float32)
HORIZON_TIME = np.array(HORIZON_TIME, dtype=np.float32)
# Turn HORIZON_FRAMES into a set, removing repeated values
HORIZON_FRAMES = set(HORIZON_FRAMES)

# "skeleton" dict: keys represent the selected structure when loading the
# dataset, while values list the body parts where to evaluate the model
SKELETON = {"full_body": ["full_body", "legs_arms", "legs", "arms"],
            "legs_arms": ["legs_arms", "legs", "arms"],
            "legs": ["legs"],
            "arms": ["arms"]}


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
    # Iterate through all selected (input) skeleton structures
    for skel_input in SKELETON.keys():
        # Iterate through all selected (prediction) skeleton structures
        for skel_pred in SKELETON[skel_input]:
            # Create NumPy arrays to store mean and stdev for each combination
            # of sampling time and horizon time
            mean = np.zeros(SAMPLING_TIME.shape)
            stdev = np.zeros(SAMPLING_TIME.shape)
            # Create another array, to accumulate the number of data points
            pts = np.zeros(SAMPLING_TIME.shape)
            # Iterate through all specified horizon frames
            for horizon_frames in iter(HORIZON_FRAMES):
                # Load validation data
                mapped_ds = parsed_ds.map(lambda x: dataset.map_dataset(
                    x, skeleton=skel_input, horizon_frames=horizon_frames),
                    num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                # Create a dataset for evaluation
                eval_ds = mapped_ds.batch(256).prefetch(tf.data.AUTOTUNE)
                # Predict for the whole dataset
                print(f"Predicting with skeleton_input={skel_input}, "
                      f"skeleton_prediction={skel_pred}, "
                      f"and horizon_frames={horizon_frames}")
                prediction = model.predict(eval_ds, verbose=1)
                # Compute the number of data points for a single recording
                rec_pts = prediction.shape[1]*prediction.shape[2]
                # Create a dataset to be used as reference
                # All sequences are joined in a single large batch
                reference_ds = (mapped_ds.batch(prediction.shape[0])
                                .prefetch(tf.data.AUTOTUNE))
                # Extract the values as NumPy arrays
                inputs, pose_targets = next(reference_ds.as_numpy_iterator())
                # Compute the absolute error between targets and predictions
                abs_err = np.abs(prediction - pose_targets)
                # Remove axes from the recordings, according to skel_pred
                if skel_pred == "legs_arms":
                    abs_err = np.delete(abs_err, (0, 1, 2, 9, 10, 11, 18, 19,
                                                  20, 27, 28, 29, 36, 37, 38,
                                                  39, 40, 41, 42, 43, 44, 45,
                                                  46, 47), axis=1)
                elif skel_pred == "legs":
                    abs_err = np.delete(abs_err, (0, 1, 2, 9, 10, 11, 18, 19,
                                                  20, 27, 28, 29), axis=1)
                    abs_err = np.delete(abs_err, np.s_[36:], axis=1)
                elif skel_pred == "arms":
                    abs_err = np.delete(abs_err, np.s_[:48], axis=1)
                # Compute mean and standard deviation for each recording
                rec_mean = np.mean(abs_err, axis=(1, 2))
                rec_stdev = np.std(abs_err, axis=(1, 2))
                # Extract sampling and horizon times from "inputs"
                sampling_input = np.round(inputs["elapsed_input"][:, 0, 0], 4)
                horizon_input = np.round(inputs["horizon_input"][:, 0, 0], 4)
                for n in range(sampling_input.shape[0]):
                    # A "horizon_input" zeroed out means an impossible shift
                    if horizon_input[n] != 0:
                        # Locate sampling and horizon time in the global array
                        pos = np.where(np.logical_and(
                            SAMPLING_TIME == sampling_input[n],
                            HORIZON_TIME == horizon_input[n]))[0]
                        # Check if this combination is desired
                        if pos.size != 0:
                            # Retrieve the index
                            i = pos.item()
                            # Compute new values of mean and standard deviation
                            new_pts = pts[i] + rec_pts
                            new_mean = (pts[i]*mean[i] +
                                        rec_pts*rec_mean[n]) / new_pts
                            new_stdev = np.sqrt((
                                pts[i]*stdev[i]**2 + rec_pts*rec_stdev[n]**2 +
                                pts[i]*rec_pts*(mean[i] - rec_mean[n])**2 /
                                new_pts) / new_pts)
                            # Update values
                            mean[i] = new_mean
                            stdev[i] = new_stdev
                            pts[i] = new_pts
            # Remove unused sampling rates and prediction horizons
            sampling_time = np.delete(SAMPLING_TIME, pts == 0)
            horizon_time = np.delete(HORIZON_TIME, pts == 0)
            mean = np.delete(mean, pts == 0)
            stdev = np.delete(stdev, pts == 0)
            # Save to a NumPy npz file
            np.savez(f"universal/skeleton_input={skel_input} "
                     f"skeleton_prediction={skel_pred}.npz",
                     sampling_time=sampling_time,
                     horizon_time=horizon_time,
                     mean=mean, stdev=stdev)
