from continual_habitat_lab.scene_manager import SceneManager
from continual_habitat_lab.config import ContinualHabitatLabConfig
import gym
import gym.spaces.dict
from habitat_sim import Simulator
from habitat_sim.logging import logger as habitat_logger
from typing import List, Union
from continual_habitat_lab.tasks.tasks import Task
from continual_habitat_lab.task_collection.task_iterator import TaskIterator
import time, logging
import numpy as np
from continual_habitat_lab.logger import chlab_logger, logging


class ContinualHabitatEnv(gym.Env):
    sim: Simulator
    task_iterator: TaskIterator
    scene_manager: SceneManager
    _episode_over: bool
    _episode_counter: int
    _action_counter: int
    _episode_start_time: float
    _episodes_since_task_change: int
    _steps_since_task_change: int
    _config: ContinualHabitatLabConfig
    _last_observation: gym.spaces.dict.Dict

    def __init__(self, config: ContinualHabitatLabConfig, verbose: int = 0) -> None:
        if not verbose:
            habitat_logger.setLevel(3)
            chlab_logger.setLevel(logging.ERROR)

        self.scene_manager = SceneManager(config)
        scene = self.scene_manager.current_scene

        self.sim = Simulator(config.make_habitat_sim_config(scene))
        # init agent(s)
        for i in range(len(config.habitat_sim_config.agents)):
            self.sim.initialize_agent(i)

        self.task_iterator = TaskIterator(config, self.sim)
        self._config = config
        self._init_bookeeping()

    def _init_bookeeping(self):
        self._episode_over = False
        self._episode_counter = 0
        self._action_counter = 0
        self._episode_start_time = None
        self._episodes_since_task_change = 0
        self._steps_since_task_change = 0

    def reset(self):
        self._episode_start_time = time.time()
        self._episode_over = False

        self._episode_counter += 1
        self._episodes_since_task_change += 1

        # task may change on new episode
        task = self._get_task(is_reset=True)

        # scene may also change on new episode
        scene, scene_changed = self.scene_manager.get_scene(self._episode_counter, self._action_counter)

        # reconfigure only on scene change
        if scene_changed:
            chlab_logger.info(f"Changing scene to {scene}..")
            self.sim.reconfigure(self._config.make_habitat_sim_config(scene))
            # reset position
            # self.sim.initialize_agent(0)
            self.current_task.on_scene_change()

        task.on_new_episode()

        self._last_observation = self.sim.reset()
        return self._last_observation

    # TODO: continuous actions spaces aren't supported with a nice api by habitat yet
    def step(self, action: Union[str, int], dt: float = 1 / 60):
        # The desired amount of time to advance the physical world.
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over == False
        ), "Episode over, call reset before calling step"

        # get current task, it may change on action taken
        task = self._get_task()
        # filter valid action, dynamically define A(s), set of allowed actions at state s
        action_key = action
        if type(action) is int:
            action_key = task.action_space_mapping(action)

        obs = self.sim.step(action_key, dt)
        # call reward function and goal test to construct observation
        reward = task.reward_function(self._last_observation, obs, action)
        self._last_observation = obs

        # call goal_test to know whether we reached final configuration
        self._episode_over = task.goal_test(obs)

        self._action_counter += 1
        self._steps_since_task_change += 1
        return obs, reward, self._episode_over, self.info

    def render(self, mode):
        return self._last_observation

    def _get_task(self, is_reset: bool = False) -> Task:
        prev_task = self.current_task
        task, changed = self.task_iterator.get_task(
            self._episodes_since_task_change, self._steps_since_task_change
        )
        if changed:
            prev_task.on_task_destroyed()
            task.on_task_change()
            if is_reset:
                self._episodes_since_task_change = 1
            else:
                self._steps_since_task_change = 1

        return task

    @property
    def tasks(self) -> List[Task]:
        return self.task_iterator.tasks

    @property
    def current_task(self) -> Task:
        return self.task_iterator.current_task

    @property
    def action_space(self):
        return self.current_task.action_space

    # @property
    # def agent_action_space(self):
    # return self._config.habitat_sim_config.agents[0].action_space

    @property
    def observation_space(self):
        # obs space only defined by sensor specification
        return

    @property
    def reward_range(self):
        return self.current_task.reward_range

    @property
    def elapsed_time(self):
        return time.time() - self._episode_start_time

    @property
    def current_scene(self):
        return self.scene_manager.current_scene

    @property
    def done(self):
        return self._episode_over

    @property
    def agent_position(self):
        return self.sim.get_agent(0).get_state().position

    @property
    def info(self):
        return {
            "episode_counter": self._episode_counter,
            "action_counter": self._action_counter,
            "elapsed_episode_time": self.elapsed_time,
            "current_scene": self.current_scene,
        }

    def seed(self, seed):
        return self.sim.seed(seed)

    def close(self):
        self.sim.close()

    # for with usage
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
