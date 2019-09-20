# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.
# Each batch element is a whole signal (from 12 sensors) registered for one trial of squeezing.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import pickle, os
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.keras.backend.set_session(tf.Session(config=config))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RNN(tf.keras.Model):

    def __init__(self):
        super(RNN, self).__init__()

        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 5, 2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Conv1D(128, 5, 2, activation=tf.nn.relu),
            # tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Conv1D(256, 5, 2, activation=tf.nn.relu)
        ])
        self.LSTM = tf.keras.layers.CuDNNLSTM(256)
        self.estimator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Dense(64, tf.nn.relu),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, None)
        ])

    def call(self, inputs, training=None, mask=None):
        cnn = self.CNN(inputs, training=training)
        lstm = self.LSTM(cnn, training=training)
        return tf.squeeze(self.estimator(lstm, training=training), 1)


def do_regression(args):
    # load train & data sets
    with open(args.data_path_train, "rb") as fp:
        train_dataset = pickle.load(fp)
        train_mean = np.mean(train_dataset["data"], axis=(0, 1), keepdims=True)
        train_var = np.var(train_dataset["data"], axis=(0, 1), keepdims=True)
    with open(args.data_path_test, "rb") as fp:
        test_dataset = pickle.load(fp)

    # load as tensorflow datasets for easy handling
    train_ds = tf.data.Dataset.from_tensor_slices((train_dataset["data"], train_dataset["stiffness"]))\
        .shuffle(50)\
        .batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_dataset["data"], test_dataset["stiffness"]))\
        .shuffle(50)\
        .batch(args.batch_size)

    # create tensorflow writers
    os.makedirs(args.results, exist_ok=True)
    train_writer, test_writer = tf.contrib.summary.create_file_writer(
        args.results), tf.contrib.summary.create_file_writer(args.results)
    train_writer.set_as_default()

    # setup model
    model = RNN()
    regularizer = tf.keras.regularizers.l2(1e-5)

    # add eta decay
    eta = tf.contrib.eager.Variable(3e-4)
    eta_value = tf.train.exponential_decay(
        3e-4,
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
                predictions = model(x_train, training=True)
                rms = tf.keras.losses.mean_squared_error(y_train, predictions)
                # reg = tf.contrib.layers.apply_regularization(regularizer, model.trainable_variables)
                # total = rms + reg
                total = rms

            gradients = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(rms)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train_loss', rms, step=n)
                train_writer.flush()
            n += 1

        # validate after each epoch
        test_writer.set_as_default()
        for x_test, y_test in test_ds:
            predictions = model((x_test - train_mean) / np.sqrt(train_var), training=False)
            rms = tf.keras.losses.mean_squared_error(y_test, predictions)
            test_loss(rms)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('test_loss', rms, step=k)
                test_writer.flush()
            k += 1

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), test_loss.result()))
        eta.assign(eta_value())

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        # save each 100 epochs
        if epoch % 100 == 0:
            ckpt_man.save()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path-train', type=str, default="./data/dataset/train_dataset.pickle")
    parser.add_argument('--data-path-test', type=str, default="./data/dataset/test_dataset.pickle")
    parser.add_argument('--results', type=str, default="./data/logs")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=5)
    args, _ = parser.parse_known_args()
    do_regression(args)
