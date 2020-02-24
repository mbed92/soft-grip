# Author: Michał Bednarek PUT Poznan
# Comment: Helper script for validating data created from the simulation

import pickle

import numpy as np

# path = "./data/dataset/final_ds/real/real_train.pickle"
# paths = ["./data/dataset/final_ds/real/real_train.pickle",
#          "./data/dataset/final_ds/real/real_val.pickle",
#          "./data/dataset/final_ds/real/real_test.pickle"]

# paths = ["./data/dataset/final_ds/sim/sim_train.pickle",
#          "./data/dataset/final_ds/sim/sim_val.pickle"]

paths = ["./data/dataset/40_10_60/",
         "./data/dataset/final_ds/real/real_val.pickle",
         "./data/dataset/final_ds/real/real_test.pickle"]


def noised_modality(data, noise_mag: float = 0.2):
    noise = np.random.uniform(-noise_mag, noise_mag, size=data.shape)
    data += noise
    return data


def playground():
    labels, samples = list(), list()
    for path in paths:
        with open(path, "rb") as fp:
            ds = pickle.load(fp)
            labels.append(ds["stiffness"])
            samples.append(ds["data"])

    labels = np.concatenate([*labels], axis=0)
    samples = np.concatenate([*samples], axis=0)

    values = np.unique(labels)
    train_dataset_x, train_dataset_y = list(), list()
    val_dataset_x, val_dataset_y = list(), list()
    test_dataset_x, test_dataset_y = list(), list()

    for i, val in enumerate(values):
        arr = np.where(labels == val, 1, 0)
        idx = np.argwhere(arr == 1).flatten()

        idx_train, idx_val, idx_test = idx[:30], idx[30:40], idx[40:100]

        # samples split
        x_train, y_train = samples[idx_train, :, :], labels[idx_train]
        train_dataset_x.append(x_train)
        train_dataset_y.append(y_train)

        x_val, y_val = samples[idx_val, :, :], labels[idx_val]
        val_dataset_x.append(x_val)
        val_dataset_y.append(y_val)

        x_test, y_test = samples[idx_test, :, :], labels[idx_test]
        test_dataset_x.append(x_test)
        test_dataset_y.append(y_test)

        print("Val: {}, num_samples: {}".format(val, arr.sum()))

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

    # for acc, stif in zip(data["data"], data["stiffness"]):
    #     acc1 = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
    #     acc2 = np.sqrt(acc[:, 3] ** 2 + acc[:, 4] ** 2 + acc[:, 5] ** 2)
    #     gyr1 = np.sqrt(acc[:, 6] ** 2 + acc[:, 7] ** 2 + acc[:, 8] ** 2)
    #     gyr2 = np.sqrt(acc[:, 9] ** 2 + acc[:, 10] ** 2 + acc[:, 11] ** 2)
    #     mag = [acc1, acc2, gyr1, gyr2]
    #
    #     # acc
    #     plt.subplot(4, 1, 1)
    #     plt.plot(mag[0], 'r')
    #     plt.subplot(4, 1, 2)
    #     plt.plot(mag[1], 'g')
    #     plt.subplot(4, 1, 3)
    #     plt.plot(mag[2], 'b')
    #     plt.subplot(4, 1, 4)
    #     plt.plot(mag[3], 'y')
    #
    #     plt.show()
    #     input(stif)


if __name__ == '__main__':
    playground()
