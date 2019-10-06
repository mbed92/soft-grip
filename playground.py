# Author: Michał Bednarek PUT Poznan
# Comment: Helper script for validating data created from the simulation

import numpy as np
import matplotlib.pyplot as plt
from create_dataset import NUM_EPISODES, MAX_ITER_PER_EP
import pickle

path = "./data/dataset/test/test_dataset.pickle"


# def playground():
#     data = np.loadtxt(path)
#     mean_acc, var_acc, force = list(), list(), list()
#     for i in range(NUM_EPISODES):
#         mean_acc.append(np.mean(data[i*MAX_ITER:i*MAX_ITER+MAX_ITER, :-1]))
#         var_acc.append(np.var(data[i*MAX_ITER:i*MAX_ITER+MAX_ITER, :-1]))
#         force.append((data[i*MAX_ITER+1, -1]))
#
#     mean_acc, var_acc, force = np.asarray(mean_acc), np.asarray(var_acc), np.asarray(force)
#     print(mean_acc, var_acc, force)
#
#     plt.subplot(3, 1, 1)
#     plt.plot(mean_acc, 'o-')
#     plt.subplot(3, 1, 2)
#     plt.plot(var_acc, '.-')
#     plt.subplot(3, 1, 3)
#     plt.plot(force, '*-')
#     plt.show()

def playground():
    with open(path, "rb") as fp:
        data = pickle.load(fp)

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

        # # gyro
        # plt.subplot(7, 1, 5)
        # plt.plot(acc[:, 12], 'r')
        # plt.subplot(7, 1, 6)
        # plt.plot(acc[:, 13], 'g')
        # plt.subplot(7, 1, 7)
        # plt.plot(acc[:, 14], 'b')
        plt.show()
        input(stif)


if __name__ == '__main__':
    playground()
