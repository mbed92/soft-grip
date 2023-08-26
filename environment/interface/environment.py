class Env(object):
    def __init__(self, sim_start, sim_step):
        self.sim_start = sim_start
        self.sim_step = sim_step

    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def reset(self):
        raise NotImplementedError("Not implemented")
