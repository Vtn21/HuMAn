"""train.py

Script for training the HuMAn model.
Use command line arguments to specify the target behavior.

Author: Victor T. N.
"""


import argparse
import glob
import os
import sys
from box import Box
from human.model.human import HuMAn
from human.model.train import full_training_loop
from human.utils import dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


# Hyperparameter configuration
common = Box({"tfr_home": "../../AMASS/tfrecords",
              "ckpt_dir": "checkpoints",
              "log_dir": "logs",
              "save_dir": "saves",
              "adam_lr": 1e-3,
              "sgd_lr": 1e-5})
train_universal = Box({"name": "train_universal",
                       "seq_len":  [256, 256, 256, 512, 512, 1024, 256],
                       "batch_size": [8,  16,  32,   8,  16,    8,  32],
                       "swa":        [0,   0,   0,   0,   0,    0,   1],
                       "rlrp":       [0,   0,   1,   0,   1,    1,   0],
                       "patience":   [0,   0,   1,   0,   1,    1,   2]})
train_bmlhandball = Box({"name": "train_bmlhandball",
                         "seq_len":  [256, 256, 256, 512, 512, 256],
                         "batch_size": [8,  16,  32,   8,  16,  32],
                         "swa":        [0,   0,   0,   0,   0,   1],
                         "rlrp":       [0,   0,   1,   0,   1,   0],
                         "patience":   [0,   0,   1,   0,   1,   2]})
transfer_bmlhandball = Box({"name": "transfer_bmlhandball",
                            "seq_len":   [256, 512, 256],
                            "batch_size": [32,  16,  32],
                            "swa":         [0,   0,   1],
                            "rlrp":        [1,   1,   0],
                            "patience":    [1,   1,   2]})
train_mpihdm05 = Box({"name": "train_mpihdm05",
                      "seq_len":  [256, 256, 256, 512, 512, 1024, 256],
                      "batch_size": [8,  16,  32,   8,  16,    8,  32],
                      "swa":        [0,   0,   0,   0,   0,    0,   1],
                      "rlrp":       [0,   0,   1,   0,   1,    1,   0],
                      "patience":   [0,   0,   1,   0,   1,    1,   2],
                      "subjects": ["bk", "dg", "mm", "tr"]})
transfer_mpihdm05 = Box({"name": "transfer_mpihdm05",
                         "seq_len":   [256, 512, 1024, 256],
                         "batch_size": [32,  16,    8,  32],
                         "swa":         [0,   0,    0,   1],
                         "rlrp":        [1,   1,    1,   0],
                         "patience":    [1,   1,    1,   2],
                         "subjects": ["bk", "dg", "mm", "tr"]})


if __name__ == "__main__":
    # Receive tuning arguments from command line
    parser = argparse.ArgumentParser(
        description="Train the HuMAn neural network.")
    parser.add_argument("-d", type=str, default="universal",
                        choices=["universal", "bmlhandball", "mpihdm05"],
                        help="Dataset (universal, bmlhandball, or mpihdm05)")
    parser.add_argument("-p", type=str, default="train",
                        choices=["train", "transfer"], help="Procedure "
                        "(training from scratch or transfer learning)")
    args = parser.parse_args()
    # Select the corresponding configuration
    if args.d == "bmlhandball":
        if args.p == "train":
            cfg = train_bmlhandball
        else:
            cfg = transfer_bmlhandball
    elif args.d == "mpihdm05":
        if args.p == "train":
            cfg = train_mpihdm05
        else:
            cfg = transfer_mpihdm05
    else:
        if args.p == "train":
            cfg = train_universal
        else:
            print("The universal dataset does not support transfer learning.")
            sys.exit()
    # Build a dataset for adapting the normalization layer
    norm_folder = os.path.join(common.tfr_home, "train_256")
    norm_ds = (dataset.folder_to_dataset(norm_folder)
               .map(dataset.map_pose_input,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
               .batch(cfg.batch_size[-1])
               .prefetch(tf.data.AUTOTUNE))
    # Instantiate the HuMAn neural network
    model = HuMAn(norm_dataset=norm_ds)
    if args.d == "universal":
        # Try to automatically restore weights from a previous checkpoint
        # (the universal training is lengthy, this allows resuming it)
        latest_ckpt = tf.train.latest_checkpoint(common.ckpt_dir)
        # TODO: finish implementation
        raise NotImplementedError("This routine is yet to be implemented.")
    else:
        if args.p == "transfer":
            # Load the universal model, for transfer learning
            universal_path = os.path.join(common.save_dir,
                                          train_universal.name)
            print(f"Loading universal model from {universal_path}")
            model.load_weights(universal_path)
        if args.d == "bmlhandball":
            # 10-fold cross-validation
            for fold in range(10):
                # Check if this fold was alredy trained
                name = f"{cfg.name}_fold{fold}"
                save_file = os.path.join(common.save_dir, f"{name}*")
                if glob.glob(save_file) != []:
                    # Save file exists, thus skip to next fold
                    continue
                else:
                    # Train using this fold
                    # Create the datasets
                    train_datasets = []
                    valid_datasets = []
                    for seq_len in cfg.seq_len:
                        # List all corresponding TFRecord files
                        tfr_list = glob.glob(os.path.join(
                            common.tfr_home, f"BMLhandball_{seq_len}", "*"))
                        # Pop one TFRecord to serve as validation
                        valid_datasets.append(dataset.tfrecords_to_dataset(
                            tfr_list.pop(fold)))
                        # Use the others as training
                        train_datasets.append(dataset.tfrecords_to_dataset(
                            tfr_list))
                    full_training_loop(model,
                                       train_datasets=train_datasets,
                                       valid_datasets=valid_datasets,
                                       seq_lengths=cfg.seq_len,
                                       batch_sizes=cfg.batch_size,
                                       swa=cfg.swa,
                                       rlrp=cfg.rlrp,
                                       patience=cfg.patience,
                                       name=name,
                                       ckpt_dir=common.ckpt_dir,
                                       log_dir=common.log_dir,
                                       save_dir=common.save_dir)
        elif args.d == "mpihdm05":
            # 4 subjects
            for subject in cfg.subjects:
                # Check if this subject was already trained
                name = f"{cfg.name}_{subject}"
                save_file = os.path.join(common.save_dir, f"{name}*")
                if glob.glob(save_file) != []:
                    # Save file exists, thus skip to next subject
                    continue
                else:
                    # Train using this subject
                    # Create the datasets
                    train_datasets = []
                    valid_datasets = []
                    for seq_len in range(len(cfg.seq_len)):
                        # Training
                        tfr_train = os.path.join(
                            common.tfr_home, f"MPI_HDM05_{seq_len}",
                            f"{subject}_train.tfrecord")
                        train_datasets.append(dataset.tfrecords_to_dataset(
                            tfr_train))
                        # Validation
                        tfr_valid = os.path.join(
                            common.tfr_home, f"MPI_HDM05_{seq_len}",
                            f"{subject}_valid.tfrecord")
                        valid_datasets.append(dataset.tfrecords_to_dataset(
                            tfr_valid))
                    full_training_loop(model,
                                       train_datasets=train_datasets,
                                       valid_datasets=valid_datasets,
                                       seq_lengths=cfg.seq_len,
                                       batch_sizes=cfg.batch_size,
                                       swa=cfg.swa,
                                       rlrp=cfg.rlrp,
                                       patience=cfg.patience,
                                       name=name,
                                       ckpt_dir=common.ckpt_dir,
                                       log_dir=common.log_dir,
                                       save_dir=common.save_dir)




