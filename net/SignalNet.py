import tensorflow as tf

from .layers import create_signal_network


class SignalNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(SignalNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, _, self.fc = create_signal_network(batch_size,
                                                        num_outputs=1,
                                                        conv_filters=[128, 128, 128],
                                                        conv_kernels=[3, 3, 3],
                                                        conv_strides=[2, 2, 2],
                                                        bilstm_units=[],
                                                        fc_layers=[256])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.fc(x, training=training)

        return x
