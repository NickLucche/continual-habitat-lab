import habitat_sim
from avalanche_lab.config import AvalancheConfig
from avalanche_lab.tasks.tasks import Task
import logging
from avalanche_lab.registry import registry
from typing import List, Dict
import numpy as np


class TaskCollection:
    _tasks: List[Task]
    _tasks_by_type: Dict[str, List[Task]] = {}
    _num_tasks: int = 0
    # _active_task_idxs:set = set()

    def __init__(self, tasks: List[Task]) -> None:
        for t in tasks:
            self._add_task_to_type_index(t)

        self._tasks = tasks
        self._num_tasks = len(tasks)

    @staticmethod
    def from_config(
        config: AvalancheConfig, sim: habitat_sim.Simulator
    ) -> "TaskCollection":
        # pass task configuration to task __init__ method
        return TaskCollection(
            list(map(lambda t: registry.get_task(t.type)(sim, **t), config.tasks))
        )

    def add_task(self, task: Task):
        self._tasks.append(task)
        self._add_task_to_type_index(task)
        self._num_tasks += 1

    def remove_task(self, task: Task):
        self._tasks.remove(task)
        self._add_task_to_type_index(task)
        self._num_tasks -= 1

    def _add_task_to_type_index(self, task: Task):
        ttype = task.__class__.__name__
        if ttype in self._tasks_by_type:
            self._tasks_by_type[ttype].append(task)
        else:
            self._tasks_by_type[ttype] = [task]

    def _remove_task_to_type_index(self, task: Task):
        ttype = task.__class__.__name__
        self._tasks_by_type[ttype].remove(task)

    def sample_tasks_idxs(self, n_tasks: int = 1, exclude_idxs=[]) -> Task:
        return np.random.choice(
            [i for i in range(len(self._tasks)) if i not in exclude_idxs], n_tasks,
        )

    def sample_tasks(self, n_tasks: int = 1, exclude_idxs=[]) -> Task:
        return [self._tasks[i] for i in self.sample_tasks_idxs(n_tasks, exclude_idxs)]

    @property
    def tasks(self):
        return [t for t in self._tasks]

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

    def __eq__(self, o: object) -> bool:
        if isinstance(o, TaskCollection):
            return self._tasks == o._tasks
        else:
            return self._tasks == o