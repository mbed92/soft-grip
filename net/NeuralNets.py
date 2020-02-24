import tensorflow as tf

from .layers import create_signal_network


class ConvNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(ConvNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, _, self.fc = create_signal_network(batch_size,
                                                        num_outputs=1,
                                                        conv_filters=[128, 128, 256],
                                                        conv_kernels=[3, 3, 3],
                                                        conv_strides=[2, 2, 2],
                                                        bilstm_units=[],
                                                        fc_layers=[256, 128])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.fc(x, training=training)

        return x


class ConvLstmNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(ConvLstmNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, self.lstm, self.fc = create_signal_network(batch_size,
                                                                num_outputs=1,
                                                                conv_filters=[128, 128, 128],
                                                                conv_kernels=[3, 3, 3],
                                                                conv_strides=[2, 2, 2],
                                                                bilstm_units=[64],
                                                                fc_layers=[256, 128])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.fc(x, training=training)

        return x
