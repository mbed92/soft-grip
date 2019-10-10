import tensorflow as tf
from .utils import add_to_tensorboard, optimize


def validate(model, writer, ds, mean, std, previous_steps, prefix="validation"):
    writer.set_as_default()
    mean_abs, mean_rms, stddev_abs, stddev_rms, mean_proc_abs, stddev_proc_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_elements = 0
    for x_val, y_val in ds:
        x_val, y_val = tf.cast(x_val, tf.float32), tf.cast(y_val, tf.float32)

        predictions = model((x_val - mean) / std, training=False)
        rms = tf.losses.mean_squared_error(y_val, predictions, reduction=tf.losses.Reduction.NONE)

        # gather stats
        absolute = tf.sqrt(rms)
        mean_abs += tf.reduce_mean(absolute)
        mean_rms += tf.reduce_mean(rms)
        mean_proc_abs += 100 - tf.reduce_mean(tf.abs(predictions) / y_val) * 100
        stddev_abs += tf.math.reduce_std(absolute)
        stddev_rms += tf.math.reduce_std(rms)
        stddev_proc_abs += tf.math.reduce_std(predictions / y_val)
        num_elements += 1

    add_to_tensorboard({
        "mean_rms": mean_rms,
        "mean_abs": mean_abs,
        "mean_proc_abs": mean_proc_abs}, num_elements, writer, previous_steps, prefix)
    previous_steps += 1

    return previous_steps, {
        "mean_abs:": mean_rms / num_elements,
        "mean_rms:": mean_rms / num_elements,
        "mean_proc_abs:": mean_proc_abs / num_elements,
        "stddev_abs:": stddev_abs / num_elements,
        "stddev_rms:": stddev_rms / num_elements,
        "stddev_proc_abs:": stddev_proc_abs / num_elements}


def train(model, writer, ds, mean, std, optimizer, previous_steps, prefix="train"):
    writer.set_as_default()
    mean_abs, mean_rms, stddev_abs, stddev_rms, mean_proc_abs, stddev_proc_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_elements = 0
    for x_train, y_train in ds:
        x_train, y_train = tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)

        with tf.GradientTape() as tape:
            predictions = model((x_train - mean) / std, training=True)
            rms = tf.losses.mean_squared_error(y_train, predictions, reduction=tf.losses.Reduction.NONE)

            # gather stats
            absolute = tf.sqrt(rms)
            mean_abs += tf.reduce_mean(absolute)
            mean_rms += tf.reduce_mean(rms)
            mean_proc_abs += (100 - tf.reduce_mean(tf.abs(predictions) / y_train) * 100)
            stddev_abs += tf.math.reduce_std(absolute)
            stddev_rms += tf.math.reduce_std(rms)
            stddev_proc_abs += tf.math.reduce_std(predictions / y_train)
            num_elements += 1

        optimize(optimizer, tape, rms, model.trainable_variables)
        add_to_tensorboard({
            "mean_rms": mean_rms,
            "mean_abs": mean_abs,
            "mean_proc_abs": mean_proc_abs}, num_elements, writer, previous_steps, prefix)
        previous_steps += 1

    return previous_steps, {
        "mean_abs:": mean_abs / num_elements,
        "mean_rms:": mean_rms / num_elements,
        "mean_proc_abs:": mean_proc_abs / num_elements,
        "stddev_abs:": stddev_abs / num_elements,
        "stddev_rms:": stddev_rms / num_elements,
        "stddev_proc_abs:": stddev_proc_abs / num_elements}
