from avalanche_lab.tasks import Task
import attr
from typing import List, Dict
import numpy as np


class TaskCollection:
    _tasks: List[Task]
    _tasks_by_type: Dict[str, List[Task]]
    # _active_task_idxs:set = set()

    def __init__(self, tasks: List[Task]) -> None:
        for t in tasks:
            self._add_task_to_type_index(t)

        self._tasks = tasks

    @staticmethod
    def from_config(config) -> "TaskCollection":
        pass

    def add_task(self, task: Task):
        self._tasks.append(task)
        self._add_task_to_type_index(task)

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
            [i for i in range(len(self._tasks)) if i not in exclude_idxs],
            n_tasks,
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
        return len(self._tasks)

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
    _active_task_idx: int
    _next_task_change_timestep: int = None

    # implement fixed and non-fixed timestep task change + sequential/random task sampling
    def __init__(self, task_collection: TaskCollection, config) -> None:
        self.tasks = task_collection
        # TODO: read config and set curr_task_idx

    def __next__(self):
        # check whether we need to change active task
        if self._check_change_task():
            self._change_active_task()
        return self.tasks[self._active_task_idx]

    def __iter__(self):
        return self
    
    def _check_change_task(self):
        pass

    def _change_active_task(self):
        # non-fixed timestep change
        # self._next_task_change_timestep = np.random.randint()
        pass
