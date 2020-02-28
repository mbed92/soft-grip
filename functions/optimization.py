import tensorflow as tf

from .utils import add_to_tensorboard, optimize


def noised_modality(data):
    # add accelerometer noise
    acc, gyro = data[:, :, :6], data[:, :, 6:]
    # acc += tf.random.normal(mean=0.0, stddev=0.7, shape=acc.get_shape(), dtype=data.dtype)
    acc += tf.random.uniform(minval=-0.5, maxval=0.5, shape=acc.get_shape(), dtype=data.dtype)

    # add gyro noise
    # gyro += tf.random.normal(mean=0.0, stddev=0.06, shape=gyro.get_shape(), dtype=data.dtype)
    gyro += tf.random.uniform(minval=-0.5, maxval=0.5, shape=gyro.get_shape(), dtype=data.dtype)

    return tf.concat([acc, gyro], -1)


def normalize_predictions(preds):
    p = 1100.0 * tf.nn.sigmoid(preds) + 300.0
    return tf.squeeze(p, -1)


def train(model, writer, ds, mean, std, optimizer, previous_steps, prefix="train", add_noise=False):
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]

    loss_metric = tf.keras.metrics.Mean("Loss")

    for i, (x_train, y_train) in enumerate(ds):
        if add_noise:
            x_train = noised_modality(x_train)

        x_train, y_train = tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)

        with tf.GradientTape() as tape:
            predictions = model((x_train - mean) / std, training=True)
            predictions = normalize_predictions(predictions)

            vars = model.trainable_variables
            l2_reg = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars]) * 0.001
            loss = tf.losses.mean_absolute_error(y_train, predictions) + l2_reg
            loss = tf.reduce_mean(loss)
            loss_metric.update_state(loss.numpy())

            loss = tf.keras.losses.mean_absolute_error(y_train, predictions)
            loss = tf.reduce_mean(loss)
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

        for m in metrics + [loss_metric]:
            m.reset_states()

        previous_steps += 1
    return previous_steps


def _val(model, writer, ds, mean, std, previous_steps=None, best_metric=None, prefix="validation", is_print=True):
    save_model = False

    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="RootMeanSquaredError"),
        tf.keras.metrics.MeanAbsoluteError(name="MeanAbsoluteError"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="MeanAbsolutePercentageError")
    ]

    save_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="save_metric")
    mae = tf.keras.metrics.MeanAbsoluteError(name="mae")
    loss_metric = tf.keras.metrics.Mean("Loss")

    for x_val, y_val in ds:
        x_val, y_val = tf.cast(x_val, tf.float32), tf.cast(y_val, tf.float32)

        predictions = model((x_val - mean) / std, training=False)
        predictions = normalize_predictions(predictions)

        loss = tf.losses.mean_squared_error(y_val, predictions)
        loss_metric.update_state(loss.numpy())

        # gather stats
        save_metric.update_state(y_val, predictions)
        mae.update_state(y_val, predictions)
        for m in metrics:
            m.update_state(y_val, predictions)

    if writer is not None and previous_steps is not None:
        with writer.as_default():
            add_to_tensorboard({
                "metrics": metrics + [loss_metric]
            }, previous_steps, prefix)
            writer.flush()
        previous_steps += 1

    if best_metric is not None:
        if save_metric.result().numpy() < best_metric:
            save_model = True
            best_metric = save_metric.result().numpy()

        if is_print:
            print("Current best test result MAE/MAPE: {} / {}".format(mae.result().numpy(), best_metric))

    for m in metrics + [loss_metric]:
        m.reset_states()

    save_metric.reset_states()
    mae.reset_states()

    return previous_steps, best_metric, save_model


def validate(model, writer, ds, mean, std, previous_steps, best_metric=None, prefix="validation"):
    save_model = False
    if type(ds) is list and type(best_metric) is list and len(best_metric) == len(ds) and len(best_metric) > 1:
        print("\n")
        for i, (sub_ds, metric) in enumerate(zip(ds, best_metric)):
            _, best, _ = _val(model, None, sub_ds, mean, std, None, metric, prefix, is_print=True)
            best_metric[i] = best
        print("\n")
    else:
        previous_steps, best_metric, save_model = _val(model, writer, ds, mean, std, previous_steps, best_metric, prefix)

    return previous_steps, best_metric, save_model
