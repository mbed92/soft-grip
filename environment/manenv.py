from .interface import Env
import mujoco_py
import numpy as np


class ManEnv(Env):
    # id of joints used to create an objects - they'll be randomized during experiments
    joint_ids = list(range(34, 251))
    tendon_ids = list(range(1))

    # list of bodies that are check for collision (partial names are enough)
    finger_names = ['g11', 'g12', 'g13', 'g2']
    obj_name = 'OBJ'

    def __init__(self, sim_start, sim_step, env_paths, is_vis=True):
        super().__init__(sim_start, sim_step)

        # setup environment and viewer
        assert len(env_paths) > 0
        self.is_vis = is_vis
        self.env_paths = env_paths
        scene = mujoco_py.load_model_from_path(env_paths[0])
        self.env = mujoco_py.MjSim(scene)
        if self.is_vis:
            self.viewer = mujoco_py.MjViewer(self.env)
        self.is_closing = True

    def load_env(self, num):
        if num < len(self.env_paths):
            new_scene = self.env_paths[num]
            scene = mujoco_py.load_model_from_path(new_scene)
            self.env = mujoco_py.MjSim(scene)
            if self.is_vis:
                self.viewer = mujoco_py.MjViewer(self.env)
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

        return np.array(self.get_sensor_sensordata()).reshape(-1)

    def reset(self):
        current_stiffness = self.set_new_stiffness()
        self.env.reset()
        self.env.forward()

        if self.sim_start > 0:
            self.step(self.sim_start)

        return current_stiffness

    def get_sensor_sensordata(self):
        data = self.env.data

        # return true when all fingers can contact an object's body
        is_contact_between_fingers_and_object = False
        fingers_left = self.finger_names
        for coni in range(data.ncon):
            contact = data.contact[coni]
            body1_name = self.env.model.geom_id2name(contact.geom1)
            body2_name = self.env.model.geom_id2name(contact.geom2)
            if body1_name is not None and body2_name is not None:
                if self.obj_name in body1_name or self.obj_name in body2_name:
                    for finger_name in fingers_left:
                        is_finger_contact = bool(finger_name in body1_name or finger_name in body2_name)
                        if is_finger_contact:
                            fingers_left.remove(finger_name)
            if len(fingers_left) == 0:
                is_contact_between_fingers_and_object = True
                break

        return data.sensordata, is_contact_between_fingers_and_object

    def toggle_grip(self):
        if self.is_closing:
            self.is_closing = False
            self._loose_hand()
        else:
            self._close_hand()
            self.is_closing = True

    def _close_hand(self):
        for i in range(4):
            self.env.data.ctrl[i] = -1.0

    def _loose_hand(self):
        for i in range(4):
            self.env.data.ctrl[i] = 0.0

    def set_new_stiffness(self, range_min=1e-1, range_max=10.0):
        new_value = np.random.uniform(range_min, range_max)
        for i in self.joint_ids:
            self.env.model.jnt_stiffness[i] = new_value
        for i in self.tendon_ids:
            self.env.model.tendon_stiffness[i] = new_value
        return new_value

    def get_env(self):
        return self.env

    def render(self):
        if self.is_vis:
            self.viewer.render()

    # specs
    @staticmethod
    def get_std_spec(args):
        return {
            "sim_start": args.sim_start,
            "sim_step": args.sim_step,
            "env_paths": args.mujoco_model_paths,
            "is_vis": args.vis
        }
