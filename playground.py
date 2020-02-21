# Author: Michał Bednarek PUT Poznan
# Comment: Helper script for validating data created from the simulation

import pickle

import matplotlib.pyplot as plt
import numpy as np

path = "./data/dataset/final_ds/mix/mix_ds_test.pickle"


def noised_modality(data, noise_mag: float = 0.2):
    noise = np.random.uniform(-noise_mag, noise_mag, size=data.shape)
    data += noise
    return data


def playground():
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    data["data"] = data["data"][:, 25:175, :]

    for acc, stif in zip(data["data"], data["stiffness"]):
        acc1 = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
        acc2 = np.sqrt(acc[:, 3] ** 2 + acc[:, 4] ** 2 + acc[:, 5] ** 2)
        gyr1 = np.sqrt(acc[:, 6] ** 2 + acc[:, 7] ** 2 + acc[:, 8] ** 2)
        gyr2 = np.sqrt(acc[:, 9] ** 2 + acc[:, 10] ** 2 + acc[:, 11] ** 2)
        mag = [acc1, acc2, gyr1, gyr2]

        # acc
        plt.subplot(4, 1, 1)
        plt.plot(mag[0], 'r')
        plt.subplot(4, 1, 2)
        plt.plot(mag[1], 'g')
        plt.subplot(4, 1, 3)
        plt.plot(mag[2], 'b')
        plt.subplot(4, 1, 4)
        plt.plot(mag[3], 'y')

        plt.show()
        input(stif)


if __name__ == '__main__':
    playground()
