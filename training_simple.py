# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import pickle, os
from tqdm import tqdm
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

    unseen = None
    if args.data_path_unseen is not None:
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
        unseen_ds = tf.data.Dataset.from_tensor_slices((unseen["data"], unseen["stiffness"]))\
            .batch(args.batch_size)

    # create tensorflow writers
    os.makedirs(args.results, exist_ok=True)
    train_writer, test_writer = tf.contrib.summary.create_file_writer(
        args.results), tf.contrib.summary.create_file_writer(args.results)
    train_writer.set_as_default()

    # setup model
    model = RNN(args.batch_size)
    regularizer = tf.keras.regularizers.l2(1e-5)

    # add eta decay
    eta = tf.contrib.eager.Variable(1e-3)
    eta_value = tf.train.exponential_decay(
        1e-3,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.99)
    eta.assign(eta_value())

    optimizer = tf.keras.optimizers.Adam(eta)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    ckpt_man = tf.train.CheckpointManager(ckpt, args.results, max_to_keep=3)

    # start training
    n, k = 0, 0
    for epoch in tqdm(range(args.epochs)):
        train_writer.set_as_default()
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                predictions = model((x_train - train_mean) / train_std, training=True)
                rms = tf.losses.mean_squared_error(y_train, predictions, reduction=tf.losses.Reduction.NONE)
                reg = tf.contrib.layers.apply_regularization(regularizer, model.trainable_variables)
                total = tf.reduce_mean(rms) + reg

            gradients = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(rms)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train/loss', tf.reduce_mean(rms), step=n)
                tf.contrib.summary.scalar('train/absolute', tf.sqrt(tf.reduce_mean(rms)), step=n)
                train_writer.flush()
            n += 1

        # validate after each epoch
        test_writer.set_as_default()
        test_rms = list()
        for x_test, y_test in test_ds:
            predictions = model((x_test - train_mean) / train_std, training=False)
            rms = tf.losses.mean_squared_error(y_test, predictions, reduction=tf.losses.Reduction.NONE)
            test_loss(rms)
            test_rms.append(rms)

        test_rms = tf.reduce_mean(tf.concat(test_rms, 0))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('test/loss', test_rms, step=k)
            tf.contrib.summary.scalar('test/absolute', tf.sqrt(test_rms), step=k)
            test_writer.flush()
        k += 1

        # check the unseen dataset
        unseen_rms = None
        if unseen is not None and unseen_ds is not None:
            rms_list = list()
            for x, y in unseen_ds:
                predictions = model((x - train_mean) / train_std, training=False)
                rms = tf.losses.mean_squared_error(y, predictions, reduction=tf.losses.Reduction.NONE)
                rms_list.append(rms)
            unseen_rms = tf.reduce_mean(tf.concat(rms_list, 0))

        if unseen_rms is not None:
            template = 'Epoch {}\tTest ABSRMS: {}, Unseen ABSRMS: {}'
            print(template.format(epoch + 1, tf.sqrt(test_rms), tf.sqrt(unseen_rms)))
        else:
            template = 'Epoch {}\tTest ABSRMS: {}'
            print(template.format(epoch + 1, tf.sqrt(test_rms)))

        # assign eta and reset the metrics for the next epoch
        eta.assign(eta_value())
        train_loss.reset_states()
        test_loss.reset_states()

        # save each 10 epochs
        if epoch % 10 == 0:
            ckpt_man.save()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/ds_IMU_no_contact_sense_full/train_dataset.pickle")
    parser.add_argument('--data-path-test', type=str, default="./data/dataset/ds_IMU_no_contact_sense_full/test_dataset.pickle")
    parser.add_argument('--data-path-unseen', type=str, default="./data/dataset/ds_IMU_no_contact_sense_full/unseen.pickle")
    parser.add_argument('--results', type=str, default="./data/logs")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=128)
    args, _ = parser.parse_known_args()
    do_regression(args)
