# Author: Michał Bednarek PUT Poznan
# Comment: Script gor generating data from Mujoco simulation for deep learning
# models. Data saved as a pickle. Each sample is a MAX_ITER_PER_EP samples of squeezing.

from argparse import ArgumentParser
from environment import ManEnv
import numpy as np
import os
from tqdm import tqdm
import pickle

NUM_EPISODES = 5000
MAX_ITER_PER_EP = 100
OPEN_CLOSE_DIV = 40


def log_into_file(args):
    assert type(args.mujoco_model_paths) is list
    num_envs = len(args.mujoco_model_paths)
    current_env = 0

    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)

    os.makedirs(args.data_folder, exist_ok=True)
    path = os.path.join(args.data_folder, "{}.pickle".format(args.data_name))
    file = open(path, 'wb')
    data, stiffness = list(), list()

    for ep in tqdm(range(NUM_EPISODES * num_envs)):
        current_stiffness = env.reset()
        # print(current_stiffness)

        # start squeezing an object
        samples = list()
        for i in range(MAX_ITER_PER_EP):
            env.render()

            # perform squeezing or loose a hand
            if i % OPEN_CLOSE_DIV == 0:
                env.toggle_grip()

            # gather readings and mask out data when there is no contact
            readings, contact = env.step()
            if args.mask_contact and not contact:
                readings = np.zeros_like(readings)
            if readings is not None:
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
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--mask-contact', type=bool, default=False)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--data-folder', type=str, default="./data/dataset/ds_IMU_with_contact_sense_full_two_fingers")
    parser.add_argument('--data-name', type=str, default="train_dataset")
    parser.add_argument('--mujoco-model-paths', nargs="+", required=True)
    args, _ = parser.parse_known_args()
    log_into_file(args)
