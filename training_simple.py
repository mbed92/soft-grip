# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import pickle, os
from tqdm import tqdm
from net import RNN
from functions import train, validate

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

    unseen = None
    if args.data_path_unseen is not "":
        with open(args.data_path_unseen, "rb") as fp:
            unseen = pickle.load(fp)

    # load as tensorflow datasets for easy handling
    train_ds = tf.data.Dataset.from_tensor_slices((train_dataset["data"], train_dataset["stiffness"])) \
        .shuffle(50) \
        .batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_dataset["data"], test_dataset["stiffness"])) \
        .shuffle(50) \
        .batch(args.batch_size)

    unseen_ds = None
    if unseen is not None:
        unseen_ds = tf.data.Dataset.from_tensor_slices((unseen["data"], unseen["stiffness"])) \
            .batch(args.batch_size)

    # create tensorflow writers
    os.makedirs(args.results, exist_ok=True)
    train_writer, test_writer = tf.contrib.summary.create_file_writer(
        args.results), tf.contrib.summary.create_file_writer(args.results)
    train_writer.set_as_default()

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

    optimizer = tf.keras.optimizers.Adam(eta)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    ckpt_man = tf.train.CheckpointManager(ckpt, args.results, max_to_keep=3)
    if args.restore_dir is not "":
        path = tf.train.latest_checkpoint(args.restore_dir)
        ckpt.restore(path)

    pickle_path = os.path.join(args.results, 'metrics.pickle')
    metrics = open(pickle_path, 'wb')

    # start training
    n, k, u = 0, 0, 0
    train_results, val_results, unseen_results = list(), list(), list()
    for epoch in tqdm(range(args.epochs)):
        train_writer.set_as_default()
        n, t_results = train(model, train_writer, train_ds, train_mean, train_std, optimizer, n)
        train_results.append(t_results)

        # validate after each epoch
        test_writer.set_as_default()
        k, v_results = validate(model, test_writer, test_ds, train_mean, train_std, k)
        val_results.append(v_results)

        # check the unseen dataset
        if unseen is not None and unseen_ds is not None:
            u, u_results = validate(model, test_writer, unseen_ds, train_mean, train_std, u)
            unseen_results.append(u_results)

        # assign eta and reset the metrics for the next epoch
        eta.assign(eta_value())

        # save each 10 epochs
        if epoch % 10 == 0:
            ckpt_man.save()

    # dump data to the pickle
    obj = {
        "training_results": train_results,
        "validation_results": val_results
    }
    if unseen_ds is not None:
        obj['unseen_results'] = unseen_results
    pickle.dump(obj, metrics)
    metrics.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/ds_IMU_no_contact_sense_full_two_fingers_v1/train_dataset.pickle")
    parser.add_argument('--data-path-test', type=str, default="./data/dataset/ds_IMU_no_contact_sense_full_two_fingers_v1/test_dataset.pickle")
    parser.add_argument('--data-path-unseen', type=str, default="")
    parser.add_argument('--results', type=str, default="./data/logs")
    parser.add_argument('--restore-dir', type=str, default="")
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=128)
    args, _ = parser.parse_known_args()
    do_regression(args)
