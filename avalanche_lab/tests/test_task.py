from avalanche_lab.config import AvalancheConfig
from omegaconf import OmegaConf
from habitat_sim import Simulator
from avalanche_lab.tasks import VoidTask
from avalanche_lab.task_collection import TaskCollection
import pytest
from avalanche_lab.registry import registry


def get_sim():
    cfg = AvalancheConfig(from_cli=False)
    return Simulator(cfg.habitat_sim_config)


# sim = get_sim()
sim = None


def test_empty_taskc():
    tc = TaskCollection([])
    assert tc.tasks == []
    assert tc._num_tasks == 0


def test_taskc_sampling():
    tc = TaskCollection([VoidTask(sim) for _ in range(4)])
    assert [VoidTask.__name__] == list(tc._tasks_by_type.keys())
    assert len(tc._tasks_by_type[VoidTask.__name__]) == 4

    tasks = tc.sample_tasks(1, [1])
    assert tasks[0] in tc.tasks
    assert tc.tasks.index(tasks[0]) != 1
    idxs = tc.sample_tasks_idxs(10, [1])
    for i in idxs:
        assert i != 1


def test_edit_taskc():
    initt = VoidTask(sim)
    tc = TaskCollection([initt])
    t = VoidTask(sim)
    tc.add_task(t)
    tc.remove_task(t)
    assert tc.tasks == [initt]
    assert tc._num_tasks == 1


def test_from_config():
    config = {"tasks": [{"name": "MyTask", "type": "VoidTask"}]}
    config = AvalancheConfig(OmegaConf.create(config), from_cli=False)
    print("Config task", config.tasks)
    tc = TaskCollection.from_config(config, sim=None)
    assert tc._num_tasks == 1
    assert isinstance(tc.tasks[0], VoidTask)


def test_register_task():
    pass
