import tensorflow as tf


def create_bidir_lstm_layer(batch_size, lstm_units, return_sequences=False, dropout=0.3):
    forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dtype=tf.float64,
                                         dropout=dropout)
    backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                          dropout=dropout, dtype=tf.float64)
    return tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                         input_shape=(batch_size, int(2 * lstm_units)))


def _create_signal_network(batch_size, num_outputs,
                           conv_filters: list = (64, 64),
                           conv_kernels: list = (3, 3),
                           conv_strides: list = (2, 2),
                           bilstm_units: list = (64,),
                           fc_layers: list = (64,)):
    assert len(conv_strides) == len(conv_kernels) == len(conv_filters)

    # create conv1d blocks
    conv_net = tf.keras.Sequential()
    for i, (num_filters, kernel, stride) in enumerate(zip(conv_filters, conv_kernels, conv_strides)):
        conv_net.add(tf.keras.layers.Conv1D(num_filters, kernel, stride, padding="SAME"))

        if i != len(conv_filters) - 1:
            conv_net.add(tf.keras.layers.BatchNormalization())
            conv_net.add(tf.keras.layers.Activation("relu"))

    # create bilstm modules
    lstm_net = tf.keras.Sequential()
    for i, unit_size in enumerate(bilstm_units):
        return_sequences = True
        if i == len(bilstm_units) - 1:
            return_sequences = False
        lstm_net.add(create_bidir_lstm_layer(batch_size, unit_size, return_sequences=return_sequences))

    # create output layer
    fc_net = tf.keras.Sequential()
    fc_net.add(tf.keras.layers.Flatten())
    for i, fc_units in enumerate(fc_layers):
        fc_net.add(tf.keras.layers.Dense(fc_units))

        if i != len(fc_layers) - 1:
            fc_net.add(tf.keras.layers.BatchNormalization())
            fc_net.add(tf.keras.layers.Activation("relu"))

    # add number of outputs
    if num_outputs is not None and num_outputs >= 1:
        fc_net.add(tf.keras.layers.Dense(num_outputs, None))

    return conv_net, lstm_net, fc_net
