"""eval_universal.py

Evaluates the universal model, with variable skeleton structures, sampling
rates and prediction horizons. Generates a pickle file, containing mean
absolute error and standard deviation, for each combination of skeleton
structure and output joint set.

Author: Victor T. N.
"""


import os
import pickle
from human.model.human import HuMAn
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


# "skeleton" dict: keys represent the selected structure when loading the
# dataset, while values list the body parts where to evaluate the model
skeleton_dict = {"full_body": ["full_body", "legs_arms", "legs", "arms"],
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
    # Create a dict to store all absolute error measurements
    abs_err = {}
    # Iterate through all selected skeleton structures
    for skeleton in skeleton_dict.keys():
        # Start with a single future frame, and increase until total
        # dataset capacity is reached
        unable_to_shift = -1
        total_iterations = 0
        horizon_frames = 1
        while unable_to_shift < total_iterations:
            # Load validation data
            eval_ds = (parsed_ds
                       .map(lambda x: dataset.map_dataset(
                            x, skeleton=skeleton,
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
                    # Extract times
                    sampling_t = inputs["elapsed_input"].numpy().item(0)
                    prediction_t = inputs["horizon_input"].numpy().item(0)
                    # Generate a prediction
                    prediction = model(inputs)
                    # Check if this sampling time has already been used
                    if sampling_t not in abs_err.keys():
                        # Create this entry
                        abs_err[sampling_t] = {}
                    # Check if this prediction time has already been used
                    if prediction_t not in abs_err[sampling_t].keys():
                        # Initialize this dict entry
                        abs_err[sampling_t][prediction_t] = [tf.math.abs(
                            prediction - pose_target)]
                    else:
                        # Append to the list
                        abs_err[sampling_t][prediction_t].append(
                            tf.math.abs(prediction - pose_target))
                # Increment the iterations counter
                total_iterations += 1
            # Next iteration will use one more frame as prediction horizon
            horizon_frames += 1
        # Compute mean and standard deviation for each combination of
        # sampling rate and future prediction horizon
        mean = {}
        stdev = {}
        for joints in skeleton_dict[skeleton]:
            for sampling_t in abs_err.keys():
                mean[sampling_t] = {}
                stdev[sampling_t] = {}
                for prediction_t in abs_err[sampling_t].keys():
                    # Concatenate all absolute error matrices
                    concat = tf.concat(abs_err[sampling_t][prediction_t],
                                       axis=0)
                    # Remove joints that are not used
                    if joints == "legs_arms":
                        concat = tf.concat([concat[:, :, 3:9],
                                            concat[:, :, 12:18],
                                            concat[:, :, 21:27],
                                            concat[:, :, 30:36],
                                            concat[:, :, 48:]], axis=2)
                    elif joints == "arms":
                        concat = concat[:, :, 48:]
                    elif joints == "legs":
                        concat = tf.concat([concat[:, :, 3:9],
                                            concat[:, :, 12:18],
                                            concat[:, :, 21:27],
                                            concat[:, :, 30:36]], axis=2)
                    # Compute and store the mean and standard deviation
                    mean[sampling_t][prediction_t] = tf.math.reduce_mean(
                        concat).numpy().item()
                    stdev[sampling_t][prediction_t] = tf.math.reduce_std(
                        concat).numpy().item()
            # Store mean and standard deviation in a single dict
            data = {"mean": mean, "stdev": stdev}
            # Save to a pickle file
            pkl_name = f"universal/skeleton={skeleton} joints={joints}.pickle"
            with open(pkl_name, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved mean and standard deviation to {pkl_name}")
