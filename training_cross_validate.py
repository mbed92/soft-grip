# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm

from functions import *
from net import SignalNet


def do_regression(args):
    os.makedirs(args.results, exist_ok=True)

    # load & crop data
    with open(args.data_path_train, "rb") as fp:
        train_dataset = pickle.load(fp)

    with open(args.data_path_validation, "rb") as fp:
        validation_dataset = pickle.load(fp)

    with open(args.data_path_unseen, "rb") as fp:
        unseen_dataset = pickle.load(fp)

    # create one total dataset for cross validation
    total_dataset = {
        "data": np.concatenate([train_dataset["data"], validation_dataset["data"]], 0),
        "stiffness": np.concatenate([train_dataset["stiffness"], validation_dataset["stiffness"]], 0),
    }

    # get mean and stddev for standarization
    train_mean = np.mean(total_dataset["data"], axis=(0, 1), keepdims=True)
    train_std = np.std(total_dataset["data"], axis=(0, 1), keepdims=True)

    # start a cross validate training
    kf = KFold(n_splits=args.num_splits)
    for split_no, (train_idx, val_idx) in enumerate(kf.split(total_dataset["data"], total_dataset["stiffness"])):

        # save split indexes
        logs_path = os.path.join(args.results, '{}'.format(split_no))
        print("Cross-validation, split no. {}. Saving dataset samples indexes...".format(split_no))
        np.savetxt(logs_path + "{}_split_train_data_samples.txt".format(split_no), train_idx)
        np.savetxt(logs_path + "{}_split_val_data_samples.txt".format(split_no), val_idx)
        print("... saved.")

        # setup model
        model = SignalNet(args.batch_size)

        # setup optimization procedure
        eta = tf.Variable(args.lr)
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, 500, 0.99)
        eta.assign(eta_value(0))
        optimizer = tf.keras.optimizers.Adam(eta)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # restore from checkpoint
        if args.restore:
            path = tf.train.latest_checkpoint(logs_path)
            ckpt.restore(path)
        ckpt_man = tf.train.CheckpointManager(ckpt, logs_path, max_to_keep=3)

        # setup writers
        os.makedirs(logs_path, exist_ok=True)
        train_writer = tf.summary.create_file_writer(logs_path + "/train")
        val_writer = tf.summary.create_file_writer(logs_path + "/val")
        unseen_writer = tf.summary.create_file_writer(logs_path + "/test")

        # create split datasets to tf generators
        train_ds, val_ds, unseen_ds = create_tf_generators(total_dataset, unseen_dataset, train_idx, val_idx,
                                                           args.batch_size)

        # start training
        train_step, val_step, unseen_step = 0, 0, 0
        save_period = int(args.epochs * 0.2)
        for epoch in tqdm(range(args.epochs)):
            train_step = train(model, train_writer, train_ds, train_mean, train_std, optimizer, train_step, add_noise=args.add_noise)
            val_step = validate(model, val_writer, val_ds, train_mean, train_std, val_step)
            unseen_step = validate(model, unseen_writer, unseen_ds, train_mean, train_std, unseen_step,
                                   prefix="unseen")

            # assign eta
            eta.assign(eta_value(0))

            # save each save_period
            if epoch % save_period == 0:
                ckpt_man.save()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/final_ds/real/real_train.pickle")
    parser.add_argument('--data-path-validation', type=str, default="./data/dataset/final_ds/real/real_val.pickle")
    parser.add_argument('--data-path-unseen', type=str, default="data/dataset/final_ds/real/real_test.pickle")

    parser.add_argument('--results', type=str, default="data/logs/train_real_test_real")
    parser.add_argument('--restore-dir', type=str, default="data/logs/train_mix_test_real")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--num-splits', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--restore', default=True, action='store_true')
    parser.add_argument('--add-noise', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    allow_memory_growth()

    do_regression(args)
