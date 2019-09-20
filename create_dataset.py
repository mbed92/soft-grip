# Author: Michał Bednarek PUT Poznan
# Comment: Script gor generating data from Mujoco simulation for deep learning
# models. Data saved as a pickle. Each sample is a MAX_ITER_PER_EP samples of squeezing:
# in each sample first 12 numbers are XYZ readings from accelerators attached to
# fingers, last one is a stiffness of an object.

from argparse import ArgumentParser
from environment import ManEnv
import numpy as np
import os
from tqdm import tqdm
import pickle

NUM_EPISODES = 100
MAX_ITER_PER_EP = 100
# LOOSE_HAND_DIV = 100


def log_into_file(args):
    assert type(args.mujoco_model_paths) is list
    num_envs = len(args.mujoco_model_paths)
    current_env = 0

    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)

    path = os.path.join(args.data_folder, "{}.pickle".format(args.data_name))
    file = open(path, 'wb')
    data, stiffness = list(), list()

    for ep in tqdm(range(NUM_EPISODES * num_envs)):
        current_stiffness = env.reset()
        print(current_stiffness)

        # start squeezing an object
        env.close_hand()
        samples = list()
        for i in range(MAX_ITER_PER_EP):
            env.viewer.render()
            readings = env.step()
            samples.append(readings)

        # add to a pickle (important to use array(), not asarray(), because it makes a copy!)
        samples = np.array(samples)
        data.append(samples)
        stiffness.append(current_stiffness)

        # change env number
        if (ep + 1) % NUM_EPISODES == 0:
            current_env += 1
            if current_env > num_envs:
                current_env = 0
            env.load_env(current_env)

    # dump data
    pickle.dump({
        "data": data,
        "stiffness": stiffness
    }, file)
    file.close()
    print("Total number of samples: {0}".format(len(data)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sim-step', type=int, default=10)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--data-folder', type=str, default="./data/dataset")
    parser.add_argument('--data-name', type=str, default="train_dataset")
    parser.add_argument('--mujoco-model-paths', nargs="+", required=True)
    args, _ = parser.parse_known_args()
    log_into_file(args)
