# Author: Michał Bednarek PUT Poznan
# Comment: Helper script for validating data created from the simulation

import pickle

import matplotlib.pyplot as plt
import numpy as np

# path = "./data/dataset/final_ds/real/real_train.pickle"
# paths = ["./data/dataset/final_ds/real/real_train.pickle",
#          "./data/dataset/final_ds/real/real_val.pickle",
#          "./data/dataset/final_ds/real/real_test.pickle"]
#
# paths = ["./data/dataset/final_ds/sim/sim_train.pickle",
#          "./data/dataset/final_ds/sim/sim_val.pickle"]

# paths = ["./data/dataset/40_10_60/",
#          "./data/dataset/final_ds/real/real_val.pickle",
#          "./data/dataset/final_ds/real/real_test.pickle"]

# path = "data/dataset/40_10_60/real_dataset_train.pickle"
real = "./data/experiments/real_200_300/train_050.pickle"
train = "./data/experiments/sim_all/train.pickle"


def noised_modality(data, noise_mag: float = 0.2):
    noise = np.random.uniform(-noise_mag, noise_mag, size=data.shape)
    data += noise
    return data


def compute_magnitude(samples):
    return np.sqrt(samples[:, :, 0] ** 2 + samples[:, :, 1] ** 2 + samples[:, :, 2] ** 2)


def playground():
    # labels, samples = list(), list()
    # for path in paths:
    #     with open(path, "rb") as fp:
    #         ds = pickle.load(fp)
    #         labels.append(ds["stiffness"])
    #         samples.append(ds["data"])
    #
    # labels = np.concatenate([*labels], axis=0)
    # samples = np.concatenate([*samples], axis=0)
    #
    # values = np.unique(labels)
    # train_dataset_x, train_dataset_y = list(), list()
    # val_dataset_x, val_dataset_y = list(), list()
    # test_dataset_x, test_dataset_y = list(), list()
    #
    # for i, val in enumerate(values):
    #     arr = np.where(labels == val, 1, 0)
    #     idx = np.argwhere(arr == 1).flatten()
    #
    #     idx_train, idx_val, idx_test = idx[:30], idx[30:40], idx[40:100]
    #
    #     # samples split
    #     x_train, y_train = samples[idx_train, :, :], labels[idx_train]
    #     x_train[..., 0] *= -1.0
    #     x_train[..., 2] *= -1.0
    #     train_dataset_x.append(x_train)
    #     train_dataset_y.append(y_train)
    #
    #     x_val, y_val = samples[idx_val, :, :], labels[idx_val]
    #     x_val[..., 0] *= -1.0
    #     x_val[..., 2] *= -1.0
    #     val_dataset_x.append(x_val)
    #     val_dataset_y.append(y_val)
    #
    #     x_test, y_test = samples[idx_test, :, :], labels[idx_test]
    #     x_test[..., 0] *= -1.0
    #     x_test[..., 2] *= -1.0
    #     test_dataset_x.append(x_test)
    #     test_dataset_y.append(y_test)
    #
    #     print("Val: {}, num_samples: {}".format(val, arr.sum()))
    #
    # train_dataset_x = np.vstack(train_dataset_x)
    # train_dataset_y = np.vstack(train_dataset_y).flatten()
    #
    # val_dataset_x = np.vstack(val_dataset_x)
    # val_dataset_y = np.vstack(val_dataset_y).flatten()
    #
    # test_dataset_x = np.vstack(test_dataset_x)
    # test_dataset_y = np.vstack(test_dataset_y).flatten()
    #
    # file = open('data/dataset/40_10_60/real_dataset_train.pickle', 'wb')
    # pickle.dump({
    #     "data": train_dataset_x,
    #     "stiffness": train_dataset_y
    # }, file)
    # file.close()
    #
    # file = open('data/dataset/40_10_60/real_dataset_val.pickle', 'wb')
    # pickle.dump({
    #     "data": val_dataset_x,
    #     "stiffness": val_dataset_y
    # }, file)
    # file.close()
    #
    # file = open('data/dataset/40_10_60/real_dataset_test.pickle', 'wb')
    # pickle.dump({
    #     "data": test_dataset_x,
    #     "stiffness": test_dataset_y
    # }, file)
    # file.close()

    with open(train, "rb") as fp:
        sim_data = pickle.load(fp)

    with open(real, "rb") as fp:
        data_real = pickle.load(fp)

    data = {"data": np.concatenate([sim_data["data"], data_real["data"]], 0),
            "stiffness": np.concatenate([sim_data["data"], data_real["data"]], 0)}

    acc1, acc2 = list(), list()
    w1, w2 = list(), list()
    m = np.mean(data["data"], axis=(0, 1), keepdims=True)
    s = np.std(data["data"], axis=(0, 1), keepdims=True)
    # data_real["data"] = (data_real["data"] - m) / s


    for sampl, stif in zip(data_real["data"], data_real["stiffness"]):
        acc1 = np.sqrt(sampl[:, 0] ** 2 + sampl[:, 1] ** 2)
        acc2 = np.sqrt(sampl[:, 3] ** 2 + sampl[:, 4] ** 2)
        w1 = np.sqrt(sampl[:, 6] ** 2 + sampl[:, 7] ** 2)
        w2 = np.sqrt(sampl[:, 9] ** 2 + sampl[:, 10] ** 2)
        mag = [acc1, acc2, w1, w2]

        # acc
        for i, signal in enumerate(mag):
            plt.subplot(4, 1, i + 1)
            plt.plot(signal, 'r')

        plt.show()
        input(stif)
    #
    # signal = np.stack([acc1, acc2], -1)
    # file = open('data/dataset/val_acc_only_sim.pickle', 'wb')
    # pickle.dump({
    #     "data": signal,
    #     "stiffness": data["stiffness"]
    # }, file)
    # file.close()


if __name__ == '__main__':
    playground()
