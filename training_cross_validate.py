# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import pickle
import os

from sklearn.model_selection import KFold
from tqdm import tqdm
from net import RNN
from functions import *

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

    unseen = None
    if args.data_path_unseen is not "":
        with open(args.data_path_unseen, "rb") as fp:
            unseen = pickle.load(fp)

    # start a cross validate training
    kf = KFold(n_splits=args.num_splits)
    train_dataset["data"] = np.asarray(
        train_dataset["data"])  # needed as an array, because train_idx, val_idx are returned this way
    train_dataset["stiffness"] = np.asarray(train_dataset["stiffness"])
    os.makedirs(args.results, exist_ok=True)
    for split_no, (train_idx, val_idx) in enumerate(kf.split(train_dataset["data"], train_dataset["stiffness"])):
        # create pickle to dump metrics from each split
        logs_path = os.path.join(args.results, '{}'.format(split_no))
        pickle_path = os.path.join(args.results, 'metrics_{}.pickle'.format(split_no, split_no))
        metrics = open(pickle_path, 'wb')

        # setup model
        model = RNN(args.batch_size)

        # add eta decay
        eta = tf.contrib.eager.Variable(1e-3)
        eta_value = tf.train.exponential_decay(
            1e-3,
            tf.train.get_or_create_global_step(),
            args.epochs,
            0.99)
        eta.assign(eta_value())

        # setup optimizer and metrics
        optimizer = tf.keras.optimizers.Adam(eta)
        ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                   model=model,
                                   optimizer_step=tf.train.get_or_create_global_step())

        # create tensorflow writers and checkpoint saver
        ckpt_man = tf.train.CheckpointManager(ckpt, logs_path, max_to_keep=3)
        os.makedirs(logs_path, exist_ok=True)
        train_writer, val_writer, unseen_writer = tf.contrib.summary.create_file_writer(
            logs_path), tf.contrib.summary.create_file_writer(logs_path), tf.contrib.summary.create_file_writer(
            logs_path)

        # create split datasets to tf generators
        train_ds, val_ds = create_tf_generators(train_dataset, train_idx, val_idx, args.batch_size)
        unseen_ds = None
        if unseen is not None:
            unseen_ds = tf.data.Dataset.from_tensor_slices((unseen["data"], unseen["stiffness"])) \
                .batch(args.batch_size)

        # start training / add metrics to tensorboard and save results for postprocessing
        train_step, val_step, unseen_step = 0, 0, 0
        train_results, val_results, unseen_results = list(), list(), list()
        for epoch in tqdm(range(args.epochs)):
            train_step, train_metrics = train(model, train_writer, train_ds, train_mean, train_std, optimizer,
                                              train_step)
            val_step, val_metrics = validate(model, val_writer, val_ds, train_mean, train_std, val_step)

            # check the unseen dataset if needed and save to list
            train_results.append(train_metrics)
            val_results.append(val_metrics)
            if unseen is not None and unseen_ds is not None:
                unseen_step, unseen_metrics = validate(model, unseen_writer, unseen_ds, train_mean, train_std,
                                                       unseen_step, prefix="unseen")
                unseen_results.append(unseen_metrics)

            # assign eta and reset the metrics for the next epoch
            eta.assign(eta_value())

            # save each 10 epochs
            if epoch % 10 == 0:
                ckpt_man.save()

        # dump data to the pickle
        obj = {
            "split_training_results": train_results,
            "split_validation_results": val_results
        }
        if unseen_ds is not None:
            obj['split_unseen_results'] = unseen_results
        pickle.dump(obj, metrics)
        metrics.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str,
                        default="./data/dataset/final_ds/mix/mix_ds_train.pickle")
    parser.add_argument('--data-path-test', type=str,
                        default="./data/dataset/final_ds/mix/mix_ds_test.pickle")
    parser.add_argument('--data-path-unseen', type=str,
                        default="./data/dataset/final_ds/mix/mix_ds_test.pickle")
    parser.add_argument('--results', type=str, default="./data/logs/cross_validate_train_mix_test_real")
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-splits', type=int, default=4)
    args, _ = parser.parse_known_args()
    do_regression(args)
