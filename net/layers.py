import tensorflow as tf


def create_bidir_lstm_layer(batch_size, lstm_units, return_sequences=False, dropout=0.3):
    forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dtype=tf.float64,
                                         dropout=dropout)
    backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                          dropout=dropout, dtype=tf.float64)
    return tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                         input_shape=(batch_size, int(2 * lstm_units)))


def create_fc_layer(filters: list, add_last: bool, activation_fcn='relu', num_last=1):
    fc = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    for num_filters in filters:
        fc.add(tf.keras.layers.Dense(num_filters, kernel_regularizer=tf.keras.regularizers.l2()))
        fc.add(tf.keras.layers.Activation(activation_fcn))
        fc.add(tf.keras.layers.Dropout(0.3))

    if add_last:
        fc.add(tf.keras.layers.Dense(int(num_last), None, kernel_regularizer=tf.keras.regularizers.l2()))

    return fc


def create_signal_network(batch_size, num_outputs,
                          conv_filters: list = (64, 64),
                          bilstm_units: list = (64,),
                          fc_layers: list = (64,), dropout=0.3, stride=3, kernel=5):
    net = tf.keras.Sequential()

    # create conv1d blocks
    for filters in conv_filters:
        net.add(tf.keras.layers.Conv1D(filters, kernel, stride, padding="SAME"))
        net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.Activation("relu"))
        net.add(tf.keras.layers.Dropout(dropout))

    # create bilstm modules
    for unit_size in bilstm_units:
        net.add(create_bidir_lstm_layer(batch_size, unit_size))

    # create output layer
    net.add(tf.keras.layers.Flatten())
    for fc_units in fc_layers:
        net.add(tf.keras.layers.Dense(fc_units))
        net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.Activation("relu"))
        net.add(tf.keras.layers.Dropout(dropout))

    # add number of outputs
    if num_outputs is not None:
        net.add(tf.keras.layers.Dense(num_outputs, None))
    return net
