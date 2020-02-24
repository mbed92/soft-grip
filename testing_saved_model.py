# Author: Michał Bednarek PUT Poznan

import os
import pickle
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from functions import allow_memory_growth
from net import ConvNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def do_regression(args):
    with open(args.data_path_train, "rb") as fp:
        train_dataset = pickle.load(fp)

    with open(args.data_path_validation, "rb") as fp:
        validation_dataset = pickle.load(fp)

    with open(args.data_path_test, "rb") as fp:
        test_dataset = pickle.load(fp)

    # create one total dataset for cross validation
    total_dataset = {
        "data": np.concatenate([train_dataset["data"], validation_dataset["data"]], 0),
        "stiffness": np.concatenate([train_dataset["stiffness"], validation_dataset["stiffness"]], 0),
    }

    # get mean and stddev for standarization
    train_mean = np.mean(total_dataset["data"], axis=(0, 1), keepdims=True)
    train_std = np.std(total_dataset["data"], axis=(0, 1), keepdims=True)

    # setup model
    # setup model
    if args.model_type == "conv":
        model = ConvNet(args.batch_size)
    elif args.model_type == "lstm":
        model = ConvNet(args.batch_size)
    elif args.model_type == "conv_lstm":
        model = ConvNet(args.batch_size)
    else:
        model = ConvNet(args.batch_size)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.restore_path)

    # start testing
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]
    loss_metric = tf.keras.metrics.Mean("Loss")

    error = list()
    for x, y in zip(test_dataset["data"], test_dataset["stiffness"]):
        predictions = model((x - train_mean) / train_std, training=False)
        loss_metric.update_state(tf.losses.mean_absolute_error(y, predictions))

        # gather stats
        for m in metrics:
            m.update_state(y, predictions)
            if m.name == "MeanAbsolutePercentageError":
                error.append(m.result().numpy())

    # print results
    for m in metrics:
        result = m.result().numpy()
        print("{} : {}".format(m.name, result))

    # plot error/data
    plt.scatter(test_dataset["stiffness"], error)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/final_ds/sim/sim_train.pickle")
    parser.add_argument('--data-path-validation', type=str, default="./data/dataset/final_ds/sim/sim_val.pickle")
    parser.add_argument('--data-path-test', type=str, default="data/dataset/40_10_60/real_dataset_test.pickle")
    parser.add_argument('--model-type', type=str, default="conv", choices=['conv', 'lstm', 'conv_lstm'], )
    parser.add_argument('--restore-path', type=str, default="data/logs/train_sim_test_real_add_noise/0/ckpt-4")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.model_type not in ['conv', 'lstm', 'conv_lstm']:
        parser.print_help()
        sys.exit(1)

    allow_memory_growth()

    do_regression(args)
