import habitat_sim
from continual_habitat_lab.config import (
    ContinualHabitatLabConfig,
    TASK_SAMPLING,
    TASK_CHANGE_BEHAVIOR,
)
from continual_habitat_lab.tasks.tasks import Task, VoidTask
from continual_habitat_lab.logger import chlab_logger
from typing import List, Dict, Tuple
import numpy as np
from .task_collection import TaskCollection


class TaskIterator:
    tasks: TaskCollection
    _active_task_idx: int
    _next_task_change_timestep: int
    _config: ContinualHabitatLabConfig
    _task_change_behavior: TASK_CHANGE_BEHAVIOR
    _task_sampling: TASK_SAMPLING

    # implement fixed and non-fixed timestep task change + sequential/random task sampling
    def __init__(
        self, config: ContinualHabitatLabConfig, sim: habitat_sim.Simulator
    ) -> None:
        # If no tasks are preset, defaults to VoidTask
        if not len(config.tasks):
            chlab_logger.warning(
                "No task was specified in config, `VoidTask` will be set as current task"
            )
            config.tasks.append(
                config.DynamicTaskClass(name="No Task", type="VoidTask")
            )
        self.tasks = TaskCollection.from_config(config, sim)
        # set active task
        self._active_task_idx = 0
        if config.task_iterator.start_from_random_task:
            self._active_task_idx = self.tasks.sample_tasks_idxs(1)
        self._config = config
        self._task_change_behavior = self._config.task_iterator.task_change_behavior
        self._task_sampling = self._config.task_iterator.task_sampling
        # non-fixed timestep change, sample next timestep in which we change task
        if self._task_change_behavior == TASK_CHANGE_BEHAVIOR.NON_FIXED:
            self._next_task_change_timestep = np.random.randint(
                self._config.task_iterator.task_change_timesteps_low,
                self._config.task_iterator.task_change_timesteps_high,
            )

    # def __next__(self):
    def get_task(self, n_episodes: int, cumulative_steps: int) -> Tuple[Task, bool]:
        # check whether we need to change active task
        change = self._change_task(n_episodes, cumulative_steps)
        if change:
            self._change_active_task()
        return self.tasks[self._active_task_idx], change

    # def __iter__(self):
    # return self

    def _change_task(self, episode_num: int, cumulative_steps: int):
        def _check_max_task_repeat(global_counter: int, counter_threshold: int):
            # checks whether we should change task due to 'episode condition' or `step_condition`, depending
            # on the arguments passed to function
            if self._task_change_behavior == TASK_CHANGE_BEHAVIOR.FIXED:
                # change task at each fixed timestep
                if global_counter > counter_threshold:
                    return True
            elif self._task_change_behavior == TASK_CHANGE_BEHAVIOR.NON_FIXED:
                # change task after a random number of steps/episodes
                if (
                    self._next_task_change_timestep > 0
                    and global_counter > self._next_task_change_timestep
                ):
                    return True
            return False

        # don't change on first reset
        if len(self.tasks) <= 1 or episode_num == 0:
            return False

        cum_steps_change = self._config.task_iterator.get("max_task_repeat_steps", -1)
        episodes_change = self._config.task_iterator.get("max_task_repeat_episodes", -1)
        # print("Change task", episode_num, episodes_change)
        if cum_steps_change <= 0 and episodes_change <= 0:
            return False
        # episode change has priority
        elif episodes_change > 0:
            return _check_max_task_repeat(episode_num, episodes_change)
        else:
            return _check_max_task_repeat(cumulative_steps, cum_steps_change)

    def _change_active_task(self):
        if self._task_change_behavior == TASK_CHANGE_BEHAVIOR.NON_FIXED:
            # non-fixed timestep change, sample next timestep in which we change task
            self._next_task_change_timestep = np.random.randint(
                self._config.task_iterator.task_change_timesteps_low,
                self._config.task_iterator.task_change_timesteps_high,
            )

        prev_task_idx = self._active_task_idx
        if self._task_sampling == TASK_SAMPLING.SEQUENTIAL:
            self._active_task_idx = (self._active_task_idx + 1) % len(self.tasks)
        elif self._task_sampling == TASK_SAMPLING.RANDOM.name:
            # sample from other tasks with equal probability
            self._active_task_idx = self.tasks.sample_tasks_idxs(
                1, exclude_idxs=[self._active_task_idx]
            )

        chlab_logger.info(
            "Current task changed from {} (id: {}) to {} (id: {}).".format(
                self.tasks[prev_task_idx].__class__.__name__,
                prev_task_idx,
                self.tasks[self._active_task_idx].__class__.__name__,
                self._active_task_idx,
            )
        )

    @property
    def current_task(self):
        return self.tasks[self._active_task_idx]
