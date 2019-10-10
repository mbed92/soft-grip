# Author: Michał Bednarek PUT Poznan
# Comment: Just a testing script.

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

    test_ds = tf.data.Dataset.from_tensor_slices((test_dataset["data"], test_dataset["stiffness"])).batch(
        args.batch_size)

    # setup model
    model = RNN(args.batch_size)
    ckpt = tf.train.Checkpoint(model=model)
    path = tf.train.latest_checkpoint(args.results_dir)
    ckpt.restore(path)

    # start training
    k = 0
    test_rms = list()
    for x, y in test_ds:
        predictions = model((x - train_mean) / train_std, training=False)
        rms = tf.losses.mean_squared_error(y, predictions, reduction=tf.losses.Reduction.NONE)
        test_rms.append(rms)
        k += 1

    # print results
    absrms = tf.sqrt(tf.reduce_mean(tf.concat(test_rms, 0)))
    print("Unseen dataset result: {}".format(absrms))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/sim_real/train_sim_real.pickle")
    parser.add_argument('--data-path-test', type=str, default="./data/dataset/sim_real/real_test.pickle")
    parser.add_argument('--results-dir', type=str, default="./data/logs")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=128)
    args, _ = parser.parse_known_args()
    do_regression(args)
