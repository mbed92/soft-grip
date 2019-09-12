# Author: Michał Bednarek PUT Poznan
# Comment: RNN + FC network implemented in Tensorflow for regression of stiffness of the gripped object.

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.keras.backend.set_session(tf.Session(config=config))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RNN(tf.keras.Model):

    def __init__(self, input_dim):
        super(RNN, self).__init__()

        self.LSTM = tf.keras.layers.CuDNNLSTM(input_dim)
        self.estimator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,  tf.nn.relu, dtype=tf.float64, kernel_initializer=tf.keras.initializers.glorot_normal()),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, None, dtype=tf.float64, kernel_initializer=tf.keras.initializers.glorot_normal())
        ])

    def call(self, inputs, training=None, mask=None):
        inputs = inputs[tf.newaxis]
        logits = self.LSTM(inputs, training=training)
        return self.estimator(logits, training=training)


def do_regression(args):
    data = np.loadtxt(args.data_path)
    i, k = data.shape
    idx = int(0.9 * i)
    y, x = data[:idx, -1], data[:idx, :-1]
    ty, tx = data[idx:, -1], data[idx:, :-1]
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((tx, ty)).batch(args.batch_size)

    os.makedirs(args.results, exist_ok=True)
    train_writer, test_writer = tf.contrib.summary.create_file_writer(args.results), tf.contrib.summary.create_file_writer(args.results)
    train_writer.set_as_default()

    model = RNN(200)

    regularizer = tf.keras.regularizers.l2(1e-5)
    optimizer = tf.keras.optimizers.Adam(4e-5)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    ckpt_man = tf.train.CheckpointManager(ckpt, args.results, max_to_keep=3)

    eta = tf.contrib.eager.Variable(1e-5)
    eta_value = tf.train.exponential_decay(
        1e-5,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.99)
    eta.assign(eta_value())

    n, k = 0, 0
    for epoch in range(args.epochs):
        train_writer.set_as_default()
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                predictions = model(x_train)
                rms = tf.keras.losses.mean_squared_error(y_train, predictions)
                reg = tf.contrib.layers.apply_regularization(regularizer, model.trainable_variables)
                total = rms + reg
            gradients = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(rms)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train_loss', train_loss.result(), step=n)
                train_writer.flush()
            n += 1

        test_writer.set_as_default()
        for x_test, y_test in test_ds:
            predictions = model(x_test)
            rms = tf.keras.losses.mean_squared_error(y_test, predictions)
            test_loss(rms)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('test_loss', test_loss.result(), step=k)
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
    parser.add_argument('--data-path', type=str, default="./data/data.txt")
    parser.add_argument('--results', type=str, default="./data/log")
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=200)
    args, _ = parser.parse_known_args()
    do_regression(args)
