from avalanche_lab.config import AvalancheConfig, TASK_SAMPLING, TASK_CHANGE_BEHAVIOR
from avalanche_lab.tasks import Task
import logging
from avalanche_lab.registry import registry
from typing import List, Dict
import numpy as np


class TaskCollection:
    _tasks: List[Task]
    _tasks_by_type: Dict[str, List[Task]]
    _num_tasks: int
    # _active_task_idxs:set = set()

    def __init__(self, tasks: List[Task]) -> None:
        for t in tasks:
            self._add_task_to_type_index(t)

        self._tasks = tasks
        self._num_tasks = len(tasks)

    @staticmethod
    def from_config(config: AvalancheConfig) -> "TaskCollection":
        return TaskCollection(list(map(registry.get_task, config.tasks)))

    def add_task(self, task: Task):
        self._tasks.append(task)
        self._add_task_to_type_index(task)
        self._num_tasks += 1

    # def add_active_task(self, task: Task):
    #     idx = self._tasks.index(task)
    #     self._active_task_idxs.add(idx)

    def _add_task_to_type_index(self, task: Task):
        ttype = task.__class__.__name__
        if ttype in self._tasks_by_type:
            self._tasks_by_type[ttype].append(task)
        else:
            self._tasks_by_type[ttype] = [task]

    def sample_tasks_idxs(self, n_tasks: int = 1, exclude_idxs=[]) -> Task:
        return np.random.choice(
            [i for i in range(len(self._tasks)) if i not in exclude_idxs], n_tasks,
        )

    def sample_tasks(self, n_tasks: int = 1, exclude_idxs=[]) -> Task:
        return self._tasks[self.sample_tasks_idxs(n_tasks, exclude_idxs)]

    @property
    def tasks(self):
        return [t for t in self._tasks]

    # @property
    # def active_tasks(self):
    #     return self._tasks[list(self._active_task_idxs)]

    # @property
    # def inactive_tasks(self):
    #     return [t for t in self._tasks if t not in self._active_task_idxs]

    def __len__(self):
        return self._num_tasks

    def __setitem__(self, task_idx: int, task: Task):
        old_task = self._tasks[task_idx]
        self._tasks[task_idx] = task
        self._tasks_by_type[old_task.__class__.__name__].remove(old_task)

    def __getitem__(self, task_idx: int):
        return self._tasks[task_idx]

    def __iter__(self):
        for t in self._tasks:
            yield t


class TaskIterator:
    tasks: TaskCollection
    _active_task_idx: int = 0
    _next_task_change_timestep: int = None
    _config: AvalancheConfig
    _task_change_behavior: TASK_CHANGE_BEHAVIOR
    _task_sampling: TASK_SAMPLING

    # implement fixed and non-fixed timestep task change + sequential/random task sampling
    def __init__(self, config: AvalancheConfig) -> None:
        self.tasks = TaskCollection.from_config(config)
        # set active task
        if config.task_iterator.start_from_random_task:
            self._active_task_idx = self.tasks.sample_tasks_idxs(1)
        self._config = config
        self._task_change_behavior = self._config.task_iterator.task_change_behavior
        self._task_sampling = self._config.task_iterator.task_sampling

    # def __next__(self):
    def get_task(self, episode_num: int, cumulative_steps: int):
        if len(self.tasks) == 0:
            return None, False
            
        # check whether we need to change active task
        change = self._change_task(episode_num, cumulative_steps)
        if change:
            self._change_active_task(episode_num)
        return self.tasks[self._active_task_idx], change

    def __iter__(self):
        return self

    def _change_task(self, episode_num: int, cumulative_steps: int):
        def _check_max_task_repeat(global_counter: int, counter_threshold: int):
            # checks whether we should change task due to 'episode condition' or `step_condition`, depending
            # on the arguments passed to function
            if self._task_change_behavior == TASK_CHANGE_BEHAVIOR.FIXED:
                # change task at each fixed timestep
                if global_counter % counter_threshold == 0:
                    return True
            elif self._task_change_behavior == TASK_CHANGE_BEHAVIOR.NON_FIXED:
                # change task after a random number of steps/episodes
                if global_counter %  self._next_task_change_timestep == 0:
                    return True
            return False

        if len(self.tasks) <= 1 or episode_num==0:
            return False

        cum_steps_change = self._config.task_iterator.max_task_repeat_steps
        episodes_change = self._config.task_iterator.max_task_repeat_episodes

        if cum_steps_change is None and episodes_change is None:
            return False
        # episode change has priority
        elif episodes_change:
            return _check_max_task_repeat(episode_num, episodes_change)
        else:
            return _check_max_task_repeat(cumulative_steps, cum_steps_change)

    def _change_active_task(self, ep_num: int):
        if self._task_change_behavior == TASK_CHANGE_BEHAVIOR.NON_FIXED:
            # non-fixed timestep change, sample next timestep in which we change task 
            self._next_task_change_timestep = np.random.randint(
                self._config.task_iterator.task_change_timesteps_low,
                self._config.task_iterator.task_change_timesteps_high,
            ) + ep_num
        
        prev_task_idx = self._active_task_idx
        if self._task_sampling == TASK_SAMPLING.SEQUENTIAL:
            self._active_task_idx = (self._curr_task_idx + 1) % len(self.tasks)
        elif self._task_sampling == TASK_SAMPLING.RANDOM.name:
            # sample from other tasks with equal probability
            self._active_task_idx = self.tasks.sample_tasks_idxs(1, exclude_idxs=[self._active_task_idx])

        logging.info(
            "Current task changed from {} (id: {}) to {} (id: {}).".format(
                self.tasks[prev_task_idx].__class__.__name__,
                prev_task_idx,
                self.tasks[self._active_task_idx].__class__.__name__,
                self._active_task_idx,
            )
        )
        
