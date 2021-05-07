from avalanche_lab.scene_manager import SceneManager
from avalanche_lab.config import AvalancheConfig
import gym
from habitat_sim import Simulator
from typing import List
from avalanche_lab.tasks import Task, VoidTask
from avalanche_lab.task_collection import TaskCollection, TaskIterator
import time


class Env(gym.Env):
    sim: Simulator
    task_iterator: TaskIterator
    scene_manager: SceneManager
    _episode_over: bool = False
    _episode_counter: int = 0
    _action_counter: int = 0
    _episode_start_time: float = None
    _config: AvalancheConfig

    def __init__(self, config: AvalancheConfig) -> None:
        self.scene_manager = SceneManager(config)
        # TODO: refactor this
        scene, _ = self.scene_manager.get_scene(0)
        self.sim = Simulator(AvalancheConfig.habitat_sim_config(config, scene))
        # init agent
        agent = self.sim.initialize_agent(0)

        self.task_iterator = TaskIterator(config)
        self._config = config
        # TODO: check agent has required sensors to carry out tasks (obs space check)

    def reset(self):
        self._episode_start_time = time.time()
        self._episode_over = False

        self._episode_counter += 1
        # task may change on new episode
        task = self._get_task(is_reset=True)

        # scene may also change on new episode
        scene, changed = self.scene_manager.get_scene(self._episode_counter)

        # TODO: suppress output to console from sim if possible or reconfigure only on scene change
        if changed:
            self.sim.reconfigure(AvalancheConfig.habitat_sim_config(self._config, scene))
        obs = self.sim.reset()
        return obs

    def step(self, action, dt: float = 1 / 60):
        # The desired amount of time to advance the physical world.
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # get current task
        task = self._get_task()

        # filter valid action (check tasks action space)

        # call reward function and goal test to construct observation

        self._action_counter += 1

        return self.sim.step(action, dt)

    def render(self, mode):
        return super().render(mode=mode)

    def _get_task(self, is_reset: bool = False):
        task, changed = self.task_iterator.get_task(
            self._episode_counter, self._action_counter
        )

        return task

    @property
    def tasks(self):
        return self.task_iterator.tasks

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

