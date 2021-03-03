"""test.py

Author: Victor T. N.
"""

import glob
import os
from human.utils import dataset
from human.utils.visualization import view_test
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


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
