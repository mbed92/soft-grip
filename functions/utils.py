import tensorflow as tf


def create_tf_generators(train_dataset, test_dataset, train_idx, val_idx, batch_size):

    train_x = train_dataset["data"][train_idx]
    train_y = train_dataset["stiffness"][train_idx]
    num_samples = train_x.shape[0]

    val_x = train_dataset["data"][val_idx]
    val_y = train_dataset["stiffness"][val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)) \
        .shuffle(num_samples) \
        .batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_dataset["data"], test_dataset["stiffness"])).batch(batch_size)

    return train_ds, val_ds, test_ds


def add_to_tensorboard(scalars: dict, step: int, prefix: str):
    for key in scalars:
        for m in scalars[key]:
            tf.summary.scalar('{}/{}'.format(prefix, m.name), m.result().numpy(), step=step)


def optimize(optimizer, tape, loss, trainable_vars):
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)