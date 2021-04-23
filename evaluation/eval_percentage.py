"""eval_test.py

Evaluates the universal model, with fixed skeleton structure (full body),
variable sampling rate and prediction horizon. Computes, for "bins" of
prediction horizon, the following error groups:
- Top 90 %
- Top 95 %
- Average
- Worst 10 %
- Worst 5 %

Results are stored in npz files inside the "percentage" subfolder.

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
    # Split the maximum horizon time (0.5 seconds) into bins
    n_bins = 3
    bin_limits = [0, 0.5/3, 2*0.5/3, 0.5]
    # Create lists to store the absolute error values into bins
    err_bins = [[] for _ in range(n_bins)]
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
                # Locate horizon time in the specified bins
                for n in range(n_bins):
                    if bin_limits[n] <= horizon_input[i] <= bin_limits[n + 1]:
                        # Add error array to corresponding bin
                        err_bins[n].append(abs_err[i])
    # Turn lists into 1D arrays
    for n in range(n_bins):
        err_bins[n] = np.concatenate(err_bins[n], axis=None)
    # Compute metrics for each bin
    average = np.empty(n_bins)
    top90 = np.empty(n_bins)
    top95 = np.empty(n_bins)
    worst10 = np.empty(n_bins)
    worst5 = np.empty(n_bins)
    for n in range(n_bins):
        average[n] = np.mean(err_bins[n])
        kth90 = int(len(err_bins[n])*0.9)
        part90 = np.partition(err_bins[n], kth90)
        top90[n] = np.mean(part90[:kth90])
        worst10[n] = np.mean(part90[kth90:])
        kth95 = int(len(err_bins[n])*0.95)
        part95 = np.partition(err_bins[n], kth95)
        top95[n] = np.mean(part95[:kth95])
        worst5[n] = np.mean(part95[kth95:])
    # Save results
    np.savez("percentage/percentages.npz",
             bin_limits=bin_limits, average=average,
             top90=top90, top95=top95, worst10=worst10, worst5=worst5)
