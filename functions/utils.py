import numpy as np
import tensorflow as tf


def create_tf_generators(train_dataset, test_datasets, train_idx, val_idx, batch_size, real_data=None,
                         add_real_data=None):
    if train_idx is not None and val_idx is not None:
        train_x = np.array(train_dataset["data"])[train_idx.tolist()]
        train_y = np.array(train_dataset["stiffness"])[train_idx.tolist()]

        # append real data samples if needed
        if add_real_data:
            train_x = np.concatenate([train_x, real_data["data"]], 0)
            train_y = np.concatenate([train_y, real_data["stiffness"]], 0)

        val_x = np.array(train_dataset["data"])[val_idx.tolist()]
        val_y = np.array(train_dataset["stiffness"])[val_idx.tolist()]

    else:
        train_x = np.array(train_dataset["data"])
        train_y = np.array(train_dataset["stiffness"])
        val_x = np.array(real_data["data"])
        val_y = np.array(real_data["stiffness"])

    print("TRAIN NUM SAMPLES IN FOLD: {}".format(train_x.shape[0]))
    print("TRAIN VAL SAMPLES IN FOLD: {}".format(val_x.shape[0]))
    num_samples = train_x.shape[0]

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)) \
        .shuffle(num_samples) \
        .batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)

    test_ds_list = list()
    for ds in test_datasets:
        tds = tf.data.Dataset.from_tensor_slices((ds["data"], ds["stiffness"])).batch(batch_size)
        test_ds_list.append(tds)

    train_mean = np.mean(train_x, axis=(0, 1), keepdims=True)
    train_std = np.std(train_x, axis=(0, 1), keepdims=True)

    return train_ds, val_ds, test_ds_list, train_mean, train_std


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
