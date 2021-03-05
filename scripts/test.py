"""test.py

Author: Victor T. N.
"""

import glob
import os
from human.utils import dataset
from human.utils.visualization import view_test
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402

# TODO: create a new testing routine. Autoregression is no longer needed.
# Use code below as reference.

# From old test.py
if __name__ == "__main__":
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the TFRecords into datasets
    parsed_ds = dataset.load_all_splits(tfr_home)
    # Load the test data
    test_ds = (parsed_ds["test"]
               .map(dataset.map_test,
                    num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(1000)
               .batch(1)
               .prefetch(tf.data.AUTOTUNE))
    test_iter = iter(test_ds)
    # Load the trained model from checkpoint
    ckpt_path = "checkpoints/*"
    ckpt_dirs = sorted(glob.glob(ckpt_path))
    model = tf.keras.models.load_model(ckpt_dirs[-1])
    # Make a prediction
    inputs, aux = next(test_iter)
    prediction = model.predict(inputs)
    view_test(path_star="../../AMASS/models/star",
              poses_prediction=prediction[0, :-1, :],
              poses_ground_truth=inputs["pose_input"][0, 1:, :],
              betas=aux["betas"][0, :, :],
              dt=aux["dt"].numpy().item(),
              gender=tf.compat.as_str(aux["gender"].numpy().item()))


# From old "test_autoregressive.py"
if __name__ == "__main__":
    # Load the datasets
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the TFRecords into datasets
    parsed_ds = dataset.load_all_splits(tfr_home)
    # Load the test data
    test_ds = (parsed_ds["test"]
               .map(dataset.map_test,
                    num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(1000)
               .batch(1)
               .prefetch(tf.data.AUTOTUNE))
    test_iter = iter(test_ds)
    # Load the trained model from checkpoint
    ckpt_path = "checkpoints/*"
    ckpt_dirs = sorted(glob.glob(ckpt_path))
    model = tf.keras.models.load_model(ckpt_dirs[-1])
    # Split the inputs
    inputs, aux = next(test_iter)
    # model.predict(inputs)
    # Make the LSTM layer stateful
    model.get_layer("LSTM").stateful = True
    # Select a frame to start autoregressive prediction
    i_autoregressive = 100
    prediction_sequence = []
    for i in range(inputs["pose_input"].shape[1]):
        inputs_step = {"selection_input": inputs["selection_input"][0:1, i:i+1],
                       "time_input": inputs["time_input"][0:1, i:i+1]}
        if i < i_autoregressive:
            inputs_step["pose_input"] = inputs["pose_input"][0:1, i:i+1]
        else:
            inputs_step["pose_input"] = tf.reshape(prediction, [1, 1, 72])
        prediction = model.predict(inputs_step)
        prediction_sequence.append(prediction)
    poses_prediction = tf.reshape(tf.constant(prediction_sequence), (-1, 72))
    view_test(path_star="../../AMASS/models/star",
              poses_prediction=poses_prediction[:-1],
              poses_ground_truth=inputs["pose_input"][0, 1:, :],
              betas=aux["betas"][0, :, :],
              dt=aux["dt"].numpy().item(),
              gender=tf.compat.as_str(aux["gender"].numpy().item()))
    # model.get_layer("LSTM").reset_states()
