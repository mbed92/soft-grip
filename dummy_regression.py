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
            tf.keras.layers.Dense(16, tf.nn.relu, dtype=tf.float64,
                                  kernel_initializer=tf.keras.initializers.glorot_normal()),
            tf.keras.layers.Dense(1, None, dtype=tf.float64, kernel_initializer=tf.keras.initializers.glorot_normal())
        ])

    def call(self, inputs, training=None, mask=None):
        inputs = inputs[tf.newaxis]
        logits = self.LSTM(inputs, training=training)
        return self.estimator(logits, training=training)


def do_regression(args):
    data = np.loadtxt(args.log_path)
    i, k = data.shape
    idx = int(0.9 * i)
    y, x = data[:idx, -1], data[:idx, :-1]
    ty, tx = data[idx:, -1], data[idx:, :-1]
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((tx, ty)).batch(args.batch_size)

    os.makedirs(args.results, exist_ok=True)
    train_writer = tf.contrib.summary.create_file_writer(args.results)
    train_writer.set_as_default()

    model = RNN(32)
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    for epoch in range(args.epochs):
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                predictions = model(x_train)
                loss = loss_object(y_train, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

            tf.contrib.summary.scalar('train_loss', train_loss.result())
            train_writer.flush()

        for x_test, y_test in test_ds:
            predictions = model(x_test)
            t_loss = loss_object(y_test, predictions)
            test_loss(t_loss)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), test_loss.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        test_loss.reset_states()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-path', type=str, default="./log.txt")
    parser.add_argument('--results', type=str, default="./log")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    args, _ = parser.parse_known_args()
    do_regression(args)
