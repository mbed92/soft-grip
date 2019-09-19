from .interface import Env
import mujoco_py
import numpy as np


class ManEnv(Env):
    # id of joints used to create an objects - they'll be randomized during experiments
    obj_ids = list(range(34, 251))

    def __init__(self, sim_start, sim_step, env_paths):
        super().__init__(sim_start, sim_step)

        # setup environment and viewer
        assert len(env_paths) > 0
        self.env_paths = env_paths
        scene = mujoco_py.load_model_from_path(env_paths[0])
        self.env = mujoco_py.MjSim(scene)

    def load_env(self, num):
        if num < len(self.env_paths):
            scene = mujoco_py.load_model_from_path(self.env_paths[num])
            self.env = mujoco_py.MjSim(scene)
        else:
            print("Wrong number,")

    # main methods
    def step(self, num_steps=-1, actions=None, min_dist=0.1):
        if num_steps < 1:
            num_steps = self.sim_step
        try:
            for _ in range(num_steps):
                self.env.step()
        except mujoco_py.builder.MujocoException:
            self.reset()

    def reset(self):
        self.env.reset()

        if self.sim_start > 0:
            self.step(self.sim_start)

    def get_sensor_sensordata(self):
        return self.env.data.sensordata

    def close_hand(self):
        for i in range(4):
            self.env.data.ctrl[i] = -1.0

    def loose_hand(self):
        for i in range(4):
            self.env.data.ctrl[i] = 0.4

    def set_new_stiffness(self, range_min=1e-3, range_max=3.0):
        new_value = np.random.uniform(range_min, range_max)
        for i in self.obj_ids:
            self.env.model.jnt_stiffness[i] = new_value
        self.env.forward()
        return new_value

    def get_env(self):
        return self.env

    # specs
    @staticmethod
    def get_std_spec(args):
        return {
            "sim_start": args.sim_start,
            "sim_step": args.sim_step,
            "env_paths": args.mujoco_model_paths,
        }
