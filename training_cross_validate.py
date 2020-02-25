# Author: Michał Bednarek PUT Poznan

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm

from functions import *
from net import *


def do_regression(args):
    os.makedirs(args.results, exist_ok=True)

    with open(args.data_path_train, "rb") as fp:
        total_dataset = pickle.load(fp)
        print("TRAIN NUM SAMPLES: {}".format(len(total_dataset["data"])))

    with open(args.data_path_validation, "rb") as fp:
        validation_dataset = pickle.load(fp)
        print("TO-ADD NUM SAMPLES: {}".format(len(validation_dataset["data"])))

    with open(args.data_path_test, "rb") as fp:
        test_dataset = pickle.load(fp)
        print("TEST NUM SAMPLES: {}".format(len(test_dataset["data"])))

    # start a cross validate training
    kf = KFold(n_splits=args.num_splits, shuffle=True)
    for split_no, (train_idx, val_idx) in enumerate(kf.split(total_dataset["data"], total_dataset["stiffness"])):

        # save split indexes
        logs_path = os.path.join(args.results, '{}'.format(split_no))
        print("Cross-validation, split no. {}. Saving dataset samples indexes...".format(split_no))
        np.savetxt(logs_path + "{}_split_train_data_samples.txt".format(split_no), train_idx)
        np.savetxt(logs_path + "{}_split_val_data_samples.txt".format(split_no), val_idx)
        print("... saved.")

        # setup model
        if args.model_type == "conv":
            model = ConvNet(args.batch_size)
        elif args.model_type == "lstm":
            raise NotImplementedError("LSTM-only model not implemented.")
        elif args.model_type == "conv_lstm":
            model = ConvLstmNet(args.batch_size)
        else:
            model = ConvNet(args.batch_size)

        # setup optimization procedure
        eta = tf.Variable(args.lr)
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, 100, 0.99)
        eta.assign(eta_value(0))
        optimizer = tf.keras.optimizers.Adam(eta)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # restore from checkpoint
        if args.restore:
            path = tf.train.latest_checkpoint(logs_path)
            ckpt.restore(path)
        ckpt_man = tf.train.CheckpointManager(ckpt, logs_path, max_to_keep=10)

        # setup writers
        os.makedirs(logs_path, exist_ok=True)
        train_writer = tf.summary.create_file_writer(logs_path + "/train")
        val_writer = tf.summary.create_file_writer(logs_path + "/val")
        test_writer = tf.summary.create_file_writer(logs_path + "/test")

        # create split datasets to tf generators
        train_ds, val_ds, test_ds, train_mean, train_std = create_tf_generators(total_dataset, test_dataset, train_idx,
                                                                                val_idx, args.batch_size,
                                                                                real_data=validation_dataset,
                                                                                add_real_data=args.add_validation_to_train)

        # start training
        train_step, val_step, test_step = 0, 0, 0
        best_metric = 999999999.0
        for _ in tqdm(range(args.epochs)):
            train_step = train(model, train_writer, train_ds, train_mean, train_std, optimizer, train_step,
                               add_noise=args.add_noise)
            val_step, _, _ = validate(model, val_writer, val_ds, train_mean, train_std, val_step)
            test_step, best_metric, save_model = validate(model, test_writer, test_ds, train_mean, train_std, test_step,
                                                          prefix="test", best_metric=best_metric)

            # assign eta
            eta.assign(eta_value(0))

            # save each save_period
            if save_model:
                ckpt_man.save()
                print("Best MAPE model saved.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--add-validation-to-train', default=False, action='store_true')

    parser.add_argument('--data-path-train', type=str, default="data/dataset/40_10_60/real_dataset_train.pickle")
    parser.add_argument('--data-path-validation', type=str, default="data/dataset/40_10_60/real_dataset_val.pickle")
    parser.add_argument('--data-path-test', type=str, default="data/dataset/40_10_60/real_dataset_test.pickle")
    parser.add_argument('--results', type=str, default="data/logs/test_test")

    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--restore-dir', type=str, default="")

    parser.add_argument('--model-type', type=str, default="conv", choices=['conv', 'lstm', 'conv_lstm'], )

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-splits', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--add-noise', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    if args.model_type not in ['conv', 'lstm', 'conv_lstm']:
        parser.print_help()
        sys.exit(1)

    allow_memory_growth()

    print("ARGUMENTS: {}".format(args))
    do_regression(args)
