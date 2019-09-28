import tensorflow as tf


def create_tf_generators(train_dataset, train_idx, val_idx, batch_size):
    train_x = train_dataset["data"][train_idx]
    train_y = train_dataset["stiffness"][train_idx]
    val_x = train_dataset["data"][val_idx]
    val_y = train_dataset["stiffness"][val_idx]
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)) \
        .shuffle(50) \
        .batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)) \
        .shuffle(50) \
        .batch(batch_size)
    return train_ds, val_ds


def add_to_tensorboard(metrics: dict, num_elements, writer, step, prefix):
    mean_metrics = dict()
    for key in metrics:
        mean_metrics[key] = metrics[key] / num_elements

    with tf.contrib.summary.always_record_summaries():
        for key in mean_metrics:
            tf.contrib.summary.scalar('{}/{}'.format(prefix, key), mean_metrics[key], step=step)
        writer.flush()


def optimize(optimizer, tape, loss, trainable_vars):
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
