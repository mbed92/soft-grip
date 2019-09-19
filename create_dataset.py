# Author: Michał Bednarek PUT Poznan Comment: Script gor generating data from Mujoco simulation for deep learning
# models. Data saved as a TXT file. Each row is 13 numbers: first 12 are XYZ readings from accelerators attached to
# fingers, last one is a stiffness of an object.

from argparse import ArgumentParser
from environment import ManEnv
import numpy as np
import os
from tqdm import tqdm
import pickle

NUM_EPISODES = 2
MAX_ITER = 200
DIVISOR = 50


def log_into_file(args):
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)

    path = os.path.join(args.data_folder, "{}.pickle".format(args.data_name))
    file = open(path, 'wb')
    data, stiffness = list(), list()
    for _ in tqdm(range(NUM_EPISODES)):
        env.reset()
        current_stiffness = env.set_new_stiffness()

        # start squeezing an object
        env.close_hand()
        samples = list()
        for i in range(MAX_ITER):
            env.step()
            samples.append(np.asarray(env.get_sensor_sensordata()).reshape(-1))
            if i % DIVISOR == 0 and i > 0:
                env.loose_hand()
            else:
                env.close_hand()

        samples = np.asarray(samples)
        data.append(samples)
        stiffness.append(current_stiffness)

    pickle.dump({
        "data": data,
        "stiffness": stiffness
    }, file)
    file.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--data-folder', type=str, default="./data/dataset")
    parser.add_argument('--data-name', type=str, default="test_dataset")
    parser.add_argument('--mujoco-model-path', type=str,
                        default='/home/mbed/.mujoco/mujoco200/model/soft_experiments.xml')
    args, _ = parser.parse_known_args()
    log_into_file(args)
