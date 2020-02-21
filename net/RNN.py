import tensorflow as tf

from .layers import create_signal_network


class RNN(tf.keras.Model):

    def __init__(self, batch_size):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        self.signal_net = create_signal_network(batch_size, 1)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        return self.signal_net(inputs, training=training)
