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

NUM_EPISODES = 5
MAX_ITER_PER_EP = 200
LOOSE_HAND_DIV = 20


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
        env.reset()
        current_stiffness = env.set_new_stiffness()

        # start squeezing an object
        env.close_hand()
        samples = list()
        for i in range(MAX_ITER_PER_EP):
            env.step()
            readings = np.array(env.get_sensor_sensordata()).reshape(-1)
            samples.append(readings)
            if i % LOOSE_HAND_DIV == 0 and i > 0:
                env.loose_hand()
            else:
                env.close_hand()

        # add to a pickle
        samples = np.asarray(samples)
        data.append(samples)
        stiffness.append(current_stiffness)

        # change env number
        if ep % NUM_EPISODES == 0 and ep > 0:
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
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--data-folder', type=str, default="./data/dataset")
    parser.add_argument('--data-name', type=str, default="test_dataset")
    parser.add_argument('--mujoco-model-paths', nargs="+", required=True)
    args, _ = parser.parse_known_args()
    log_into_file(args)
