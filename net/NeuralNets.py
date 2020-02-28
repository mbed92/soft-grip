import tensorflow as tf

from .layers import create_signal_network


class ConvNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(ConvNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, _, self.fc = create_signal_network(batch_size,
                                                        num_outputs=1,
                                                        conv_filters=[128, 256, 512],
                                                        conv_kernels=[3, 3, 3],
                                                        conv_strides=[2, 2, 2],
                                                        bilstm_units=[],
                                                        fc_layers=[512, 256, 128, 64])
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.pooling(x, training=training)
        x = self.fc(x, training=training)

        return x


class ConvBiLstmNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(ConvBiLstmNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, self.lstm, self.fc = create_signal_network(batch_size,
                                                                num_outputs=1,
                                                                conv_filters=[128, 256, 256],
                                                                conv_kernels=[3, 3, 3],
                                                                conv_strides=[2, 2, 2],
                                                                bilstm_units=[128],
                                                                fc_layers=[512, 256, 128, 64])
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.fc(x, training=training)

        return x


class ConvLstmNet(tf.keras.Model):

    def __init__(self, batch_size):
        super(ConvLstmNet, self).__init__()

        self.batch_size = batch_size
        self.conv1d, _, self.fc = create_signal_network(batch_size,
                                                        num_outputs=1,
                                                        conv_filters=[128, 256, 256],
                                                        conv_kernels=[3, 3, 3],
                                                        conv_strides=[2, 2, 2],
                                                        bilstm_units=[],
                                                        fc_layers=[512, 256, 128, 64])

        self.lstm = tf.keras.Sequential([
            tf.keras.layers.LSTM(256, return_sequences=False, dtype=tf.float64, dropout=0.3)
        ])
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1d(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.fc(x, training=training)

        return x
