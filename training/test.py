"""test.py

Author: Victor T. N.
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from human.model.human import HuMAn
from human.utils import dataset
from human.utils.visualization import view_test
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


SHUFFLE_BUFFER = 1000


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Load the testing dataset
    parsed_ds = dataset.load_splits(tfr_home, splits=["test_1024"])
    # Load the test data
    test_ds = (parsed_ds["test_1024"]
               .map(dataset.map_test,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .shuffle(SHUFFLE_BUFFER)
               .batch(1)
               .prefetch(tf.data.AUTOTUNE))
    test_iter = iter(test_ds)
    # The HuMAn neural network
    # Expects normalization layer adapted, as model is trained
    model = HuMAn()
    # Load weights from checkpoints
    ckpt_path = "checkpoints"
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(latest_ckpt)
    # Make a prediction
    inputs, pose_target, aux = next(test_iter)
    prediction = model.predict(inputs)
    framerate = np.round(1/aux["dt"].numpy().item(), decimals=1)
    horizon = np.round(inputs['horizon_input'][0, 0].numpy().item(),
                       decimals=4)
    print(f"Input framerate is {framerate} Hz.")
    print(f"Prediction horizon is {horizon} s.")

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(prediction[0])
    axs[0].set_ylabel("Prediction")
    axs[1].plot(pose_target[0])
    axs[1].set_ylabel("Ground truth")
    axs[2].plot(np.abs(prediction[0] - pose_target[0]))
    axs[2].set_ylabel("Absolute error")
    plt.show()

    view_test(path_star="../../AMASS/models/star",
              poses_prediction=prediction[0],
              poses_ground_truth=pose_target[0],
              betas=aux["betas"][0],
              dt=aux["dt"].numpy().item(),
              gender=tf.compat.as_str(aux["gender"].numpy().item()))
