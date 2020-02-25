# Author: Michał Bednarek PUT Poznan

import os
import pickle
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from functions import allow_memory_growth, create_tf_generators
from functions.optimization import normalize_predictions
from net import ConvNet, ConvLstmNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def do_regression(args):
    with open(args.data_path_train, "rb") as fp:
        total_dataset = pickle.load(fp)
        print("TRAIN NUM SAMPLES: {}".format(len(total_dataset["data"])))

    validation_dataset = None
    if args.data_path_validation is not None:
        with open(args.data_path_validation, "rb") as fp:
            validation_dataset = pickle.load(fp)
            print("TO-ADD NUM SAMPLES: {}".format(len(validation_dataset["data"])))

    if validation_dataset is not None:
        total_dataset["data"] = np.concatenate([total_dataset["data"], validation_dataset["data"]], 0)
        total_dataset["stiffness"] = np.concatenate([total_dataset["stiffness"], validation_dataset["stiffness"]], 0)

    with open(args.data_path_test, "rb") as fp:
        test_dataset = pickle.load(fp)
        print("TEST NUM SAMPLES: {}".format(len(test_dataset["data"])))

    kf = KFold(n_splits=5, shuffle=True)
    for split_no, (train_idx, val_idx) in enumerate(kf.split(total_dataset["data"], total_dataset["stiffness"])):

        # setup model
        if args.model_type == "conv":
            model = ConvNet(args.batch_size)
        elif args.model_type == "lstm":
            raise NotImplementedError("LSTM-only model not implemented.")
        elif args.model_type == "conv_lstm":
            model = ConvLstmNet(args.batch_size)
        else:
            model = ConvNet(args.batch_size)

        # restore from checkpoint
        ckpt = tf.train.Checkpoint(model=model)
        path = tf.train.latest_checkpoint(args.restore_path)
        ckpt.restore(path)

        _, _, test_ds, train_mean, train_std = create_tf_generators(total_dataset, test_dataset, train_idx,
                                                                    val_idx, args.batch_size,
                                                                    real_data=validation_dataset,
                                                                    add_real_data=True)

        # start testing
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
        ]

        loss_metric = tf.keras.metrics.Mean("Loss")

        error = list()
        for x_train, y_train in test_ds:

            x_train, y_train = tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)
            predictions = model((x_train - train_mean) / train_std, training=False)
            predictions = normalize_predictions(predictions)
            loss_metric.update_state(tf.losses.mean_absolute_error(y_train, predictions))

            # gather stats
            for m in metrics:
                m.update_state(y_train, predictions)
                if m.name == "MeanAbsolutePercentageError":
                    error.append(m.result().numpy())

        # print results
        for m in metrics:
            result = m.result().numpy()
            print("{} : {}".format(m.name, result))

        # plot error/data
        plt.scatter(test_dataset["stiffness"], error)
        plt.show()
        break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="data/dataset/final_ds/sim/sim_train.pickle")
    parser.add_argument('--data-path-validation', type=str,
                        default="data/dataset/40_10_60/real_dataset_train_200.pickle")
    parser.add_argument('--data-path-test', type=str, default="data/dataset/testing_datasets/box_test.pickle")
    parser.add_argument('--model-type', type=str, default="conv_lstm", choices=['conv', 'lstm', 'conv_lstm'], )
    parser.add_argument('--restore-path', type=str,
                        default="data/logs/sim2rel_experiments/02_train_sim_test_real_add_noise_200_real/0")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.model_type not in ['conv', 'lstm', 'conv_lstm']:
        parser.print_help()
        sys.exit(1)

    allow_memory_growth()

    do_regression(args)
