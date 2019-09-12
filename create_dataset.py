# Author: Michał Bednarek PUT Poznan
# Comment: Script gor generating data from Mujoco simulation for deep learning models.

from argparse import ArgumentParser
from environment import ManEnv
import numpy as np

NUM_EPISODES = 100
MAX_ITER = 200
DIVISOR = 90


def log_into_file(args):
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    file = open(args.data_path, 'w')

    for _ in range(NUM_EPISODES):
        env.reset()
        current_stiffness = env.set_new_stiffness()

        # start squeezing an object
        env.close_hand()
        for i in range(MAX_ITER):
            env.step()
            data = np.asarray(env.get_sensor_sensordata()).reshape(-1)
            data = np.hstack([data, current_stiffness])
            file.write(' '.join(map(str, data)) + '\n')

            if i % DIVISOR == 0 and i > 0:
                env.loose_hand()
            else:
                env.close_hand()
    file.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--data-path', type=str, default="./data/data.txt")
    parser.add_argument('--mujoco-model-path', type=str,
                        default='/home/mbed/.mujoco/mujoco200/model/soft_experiments.xml')
    args, _ = parser.parse_known_args()
    log_into_file(args)
