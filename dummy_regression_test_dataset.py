# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import pickle, os
from net import RNN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.keras.backend.set_session(tf.Session(config=config))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def do_regression(args):
    with open(args.data_path_train, "rb") as fp:
        train_dataset = pickle.load(fp)
        train_mean = np.mean(train_dataset["data"], axis=(0, 1), keepdims=True)
        train_std = np.std(train_dataset["data"], axis=(0, 1), keepdims=True)
    with open(args.data_path_test, "rb") as fp:
        test_dataset = pickle.load(fp)

    test_ds = tf.data.Dataset.from_tensor_slices((test_dataset["data"], test_dataset["stiffness"])) \
        .shuffle(50) \
        .batch(args.batch_size)

    # setup model
    model = RNN(args.batch_size)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.results_dir)

    # start training
    n, k = 0, 0
    test_rms = list()
    for x_test, y_test in test_ds:
        predictions = model((x_test - train_mean) / train_std, training=False)
        rms = tf.losses.mean_squared_error(y_test, predictions, reduction=tf.losses.Reduction.NONE)
        test_rms.append(rms)
        k += 1

    test_rms = tf.sqrt(tf.reduce_mean(tf.concat(test_rms, 0)))
    print("Unseen dataset result: {}".format(test_rms))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/full_ds/train_dataset.pickle")
    parser.add_argument('--data-path-test', type=str, default="./data/dataset/full_ds/unseen.pickle")
    parser.add_argument('--results-dir', type=str, default="./data/logs/CNN_RNN_FC")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=128)
    args, _ = parser.parse_known_args()
    do_regression(args)
