import gym
from habitat_sim import Simulator
from typing import List
from avalanche_lab.tasks import Task, VoidTask

class Env(gym.Env):
    sim: Simulator
    tasks: List[Task]
    _episode_over: bool = False
    _episode_counter: int = 0
    _episode_start_time:float = None

    def __init__(self, sim: Simulator, tasks: List[Task]=None) -> None:
        super().__init__()
        self.sim = sim
        if tasks is None or not len(tasks):
            # set void task, it's just a placeholder which allows you to interact 
            # with the environment without any particular purpose to fulfill 
            self.tasks = [VoidTask()]

        # TODO: check agent has required sensors to carry out tasks (obs space check)

    def reset(self):
        return super().reset()

    def step(self, action, dt: float=1/60):
        # The desired amount of time to advance the physical world.
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # filter valid action (check tasks action space)

        # call reward function and goal test to construct observation

        return self.sim.step(action, dt)

    def render(self, mode):
        return super().render(mode=mode)

    @property
    def action_space(self):
        pass

    @property
    def observation_space(self):
        # TODO: merge of active tasks obs spaces. agent has bunch of sensors, but some task may require to only use a subset of them
        pass

    @property
    def reward_range(self):
        pass

    def seed(self, seed):
        return self.sim.seed(seed)

    def close(self):
        self.sim.close()

    # for with usage
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    


    # multitask env will follow later  