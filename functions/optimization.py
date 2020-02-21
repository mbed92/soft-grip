import tensorflow as tf

from .utils import add_to_tensorboard, optimize


def train(model, writer, ds, mean, std, optimizer, previous_steps, prefix="train"):
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]

    loss_metric = tf.keras.metrics.Mean("Loss")

    for x_train, y_train in ds:
        x_train, y_train = tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)

        with tf.GradientTape() as tape:
            predictions = model((x_train - mean) / std, training=True)
            loss = tf.losses.mean_squared_error(y_train, predictions)
            loss_metric.update_state(loss.numpy())

        optimize(optimizer, tape, loss, model.trainable_variables)

        # gather stats
        for m in metrics:
            m.update_state(y_train, predictions)

        with writer.as_default():
            add_to_tensorboard({
                "metrics": metrics + [loss_metric]
            }, previous_steps, prefix)
        writer.flush()

        for m in metrics:
            m.reset_states()

        previous_steps += 1
    return previous_steps


def validate(model, writer, ds, mean, std, previous_steps, prefix="validation"):
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]

    loss_metric = tf.keras.metrics.Mean("Loss")

    for x_val, y_val in ds:
        x_val, y_val = tf.cast(x_val, tf.float32), tf.cast(y_val, tf.float32)

        predictions = model((x_val - mean) / std, training=False)
        loss = tf.losses.mean_squared_error(y_val, predictions)
        loss_metric.update_state(loss.numpy())

        # gather stats
        for m in metrics:
            m.update_state(y_val, predictions)

    with writer.as_default():
        add_to_tensorboard({
            "metrics": metrics + [loss_metric]
        }, previous_steps, prefix)
        writer.flush()
    previous_steps += 1

    return previous_steps
