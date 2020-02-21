# Author: Michał Bednarek PUT Poznan
# Comment: Just a testing script.

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from net import SignalNet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.keras.backend.set_session(tf.Session(config=config))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def do_regression(args):
    # load & crop data
    with open(args.data_path_train, "rb") as fp:
        train_dataset = pickle.load(fp)
        train_dataset["data"] = train_dataset["data"]

    with open(args.data_path_unseen, "rb") as fp:
        unseen_dataset = pickle.load(fp)

    # get mean and stddev for standarization
    train_mean = np.mean(train_dataset["data"], axis=(0, 1), keepdims=True)
    train_std = np.std(train_dataset["data"], axis=(0, 1), keepdims=True)

    # setup model
    model = SignalNet(args.batch_size)
    ckpt = tf.train.Checkpoint(model=model)
    path = tf.train.latest_checkpoint(args.results_dir)
    ckpt.restore(path)

    # start testing
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]
    loss_metric = tf.keras.metrics.Mean("Loss")

    for x, y in unseen_dataset:
        predictions = model((x - train_mean) / train_std, training=False)
        loss_metric.update_state(tf.losses.mean_absolute_error(y, predictions))

        # gather stats
        for m in metrics:
            m.update_state(y, predictions)

    # print results
    for m in metrics:
        result = m.result().numpy()
        print("{} : {}".format(m.name, result))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="data/dataset/final_ds/mix/mix_ds_train.pickle")
    parser.add_argument('--data-path-test', type=str, default="data/dataset/final_ds/real/real_test.pickle")
    parser.add_argument('--results-dir', type=str, default="./data/logs")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=128)
    args, _ = parser.parse_known_args()
    do_regression(args)
