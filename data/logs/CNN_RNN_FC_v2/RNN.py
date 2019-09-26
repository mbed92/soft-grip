import tensorflow as tf


class RNN(tf.keras.Model):

    def __init__(self, batch_size):
        super(RNN, self).__init__()

        self.CNN = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 5, 2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(256, 3, 2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(512, 3, 2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3)
        ])

        forward_layer = tf.keras.layers.LSTM(512, return_sequences=False, dtype=tf.float64)
        backward_layer = tf.keras.layers.LSTM(512, activation='relu', return_sequences=False, go_backwards=True)
        self.BIDIRECTIONAL = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(batch_size, 512))
        ])

        self.FC = tf.keras.Sequential([
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, None)
        ])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        cnn = self.CNN(inputs, training=training)
        lstm = self.BIDIRECTIONAL(cnn, training=training)
        return tf.squeeze(self.FC(lstm, training=training), 1)
