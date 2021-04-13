"""eval_test.py

Evaluates the universal model, with fixed skeleton structure (full body),
variable sampling rate and prediction horizon. Computes, for each individual
prediction horizon, the following sets of mean absolute error:
- Average
- Top 95 %
- Top 90 %
- Worst 10 %
- Worst 5 %

Author: Victor T. N.
"""


import os
import pickle
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


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
    # Create a dict to store all absolute error measurements
    abs_err = {}
    # Start with a single future frame, and increase until total
    # dataset capacity is reached
    unable_to_shift = -1
    total_iterations = 0
    horizon_frames = 1
    while unable_to_shift < total_iterations:
        # Load validation data
        eval_ds = (parsed_ds
                   .map(lambda x: dataset.map_dataset(
                        x, skeleton="full_body",
                        horizon_frames=horizon_frames))
                   .batch(1).prefetch(tf.data.AUTOTUNE))
        # Create an iterator
        test_iter = iter(eval_ds)
        # Iterate through the whole dataset
        unable_to_shift = 0
        total_iterations = 0
        for inputs, pose_target in test_iter:
            if inputs["horizon_input"].numpy().all() == 0:
                # If the dataset is unable to shift the required number of
                # frames, it returns the "horizon_input" array filled with
                # zeros
                unable_to_shift += 1
            else:
                # Successfully shifted the required number of frames
                # Extract prediction time
                prediction_t = inputs["horizon_input"].numpy().item(0)
                # Generate a prediction
                prediction = model(inputs)
                # Check if this sampling time has already been used
                if prediction_t not in abs_err.keys():
                    # Create this entry
                    abs_err[prediction_t] = tf.math.abs(
                        prediction - pose_target)
                else:
                    # Append to the list
                    abs_err[prediction_t].append(
                        tf.math.abs(prediction - pose_target))
            # Increment the iterations counter
            total_iterations += 1
        # Next iteration will use one more frame as prediction horizon
        horizon_frames += 1
    # Compute percent means and standard deviations
    mean = {}
    for prediction_t in abs_err.keys():
        mean[prediction_t] = {}
        # Concatenate all absolute error matrices
        concat = tf.concat(abs_err[prediction_t], axis=0)
        # Flatten this tensor
        flat = tf.reshape(concat, [-1])
        # Sort in ascending order
        asc = tf.sort(flat)
        # Compute and store the mean and standard deviation
        mean[prediction_t]["avg"] = tf.math.reduce_mean(
            asc).numpy().item()
        mean[prediction_t]["top95"] = tf.math.reduce_mean(
            tf.slice(asc, [0], [int(tf.shape(asc).numpy().item()*0.95)]))
        mean[prediction_t]["top90"] = tf.math.reduce_mean(
            tf.slice(asc, [0], [int(tf.shape(asc).numpy().item()*0.9)]))
        mean[prediction_t]["worst10"] = tf.math.reduce_mean(
            tf.slice(asc, [int(tf.shape(asc).numpy().item()*0.1)], [-1]))
        mean[prediction_t]["worst5"] = tf.math.reduce_mean(
            tf.slice(asc, [int(tf.shape(asc).numpy().item()*0.05)], [-1]))
    # Save to a pickle file
    pkl_name = "percentage/skeleton=full_body joints=full_body.pickle"
    with open(pkl_name, "wb") as f:
        pickle.dump(mean, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved percentage means to {pkl_name}")
