from continual_habitat_lab.config import ContinualHabitatLabConfig
from omegaconf import OmegaConf
from habitat_sim import Simulator
from continual_habitat_lab.tasks.tasks import VoidTask
from continual_habitat_lab.task_collection.task_collection import TaskCollection
import pytest
from continual_habitat_lab.registry import registry
import gym

def get_sim():
    cfg = ContinualHabitatLabConfig(from_cli=False)
    return Simulator(cfg.habitat_sim_config)


# sim = get_sim()
sim = None


def test_empty_taskc():
    tc = TaskCollection([])
    assert tc.tasks == []
    assert tc._num_tasks == 0


def test_task_collection_sampling():
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
    config = ContinualHabitatLabConfig(OmegaConf.create(config), from_cli=False)
    print("Config task", config.tasks)
    tc = TaskCollection.from_config(config, sim=None)
    assert tc._num_tasks == 1
    assert isinstance(tc.tasks[0], VoidTask)

from continual_habitat_lab.registry import registry

@registry.register_task
class MoveForwardTask(VoidTask):
    def __init__(self, sim: Simulator, name: str, consecutive_fwd: int, *args, **kwargs) -> None:
        super().__init__(sim, name=name, *args, **kwargs)
        self.consecutive_forward = 0
        self.fwd_times = consecutive_fwd

    def reward_function(self, prev_obs: gym.spaces.dict.Dict, curr_obs: gym.spaces.dict.Dict, action):
        return -1

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        return self.consecutive_forward >= self.fwd_times

    def action_space_mapping(self, action: int) -> str:
        action_key = super().action_space_mapping(action)
        if action_key == 'move_forward':
            self.consecutive_forward += 1
        else:
            self.consecutive_forward = 0
        return action_key

def test_register_task():
    config = {"tasks": [{"name": "MyCustomTask", "type": "MoveForwardTask", 'consecutive_fwd': 3}]}
    config = ContinualHabitatLabConfig(OmegaConf.create(config), from_cli=False)
    print("Config task", config.tasks)
    tc = TaskCollection.from_config(config, sim=None)
    assert tc._num_tasks == 1
    assert isinstance(tc.tasks[0], MoveForwardTask)
    assert tc.tasks[0].fwd_times == 3
    task = tc.tasks[0]
    for i in range(3):
        k = task.action_space_mapping(3)
        assert k == 'move_forward'
        if i < 2:
            assert not task.goal_test(None)
        else:
            assert task.goal_test(None)


        