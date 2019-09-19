# Author: Michał Bednarek PUT Poznan
# Comment: Helper script for validating data created from the simulation

import numpy as np
import matplotlib.pyplot as plt
from create_dataset import NUM_EPISODES, MAX_ITER

path = "./data/data_softbox.txt"


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
    data = np.loadtxt(path)

    for i in range(NUM_EPISODES):

        acc = data[i * MAX_ITER:i * MAX_ITER + MAX_ITER, :-1]
        mag = list()
        for s in range(4):
            magn = acc[:, s * 3:s * 3 + 3]
            mag.append(np.sqrt(np.square(magn[:, 0]) +
                               np.square(magn[:, 1]) +
                               np.square(magn[:, 2])))

        plt.subplot(4, 1, 1)
        plt.plot(mag[0], 'r')
        plt.subplot(4, 1, 2)
        plt.plot(mag[1], 'g')
        plt.subplot(4, 1, 3)
        plt.plot(mag[2], 'b')
        plt.subplot(4, 1, 4)
        plt.plot(mag[3], 'y')
        plt.show()

        print("Iteration end: ", data[i * MAX_ITER + 1, -1])
        input("Press key to continue...")


if __name__ == '__main__':
    playground()
