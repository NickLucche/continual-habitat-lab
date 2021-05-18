from avalanche_lab.tasks import VoidTask
from omegaconf import OmegaConf
from avalanche_lab.config import AvalancheConfig, TASK_CHANGE_BEHAVIOR
from avalanche_lab.task_collection import TaskIterator
import pytest
import itertools

# make sure it works with no tasks specified in config
def test_no_tasks():
    # get default config
    cfg = AvalancheConfig(from_cli=False)
    ti = TaskIterator(cfg, None)
    assert ti.tasks == []
    task, changed = ti.get_task(0, 0)
    assert task is None and not changed


def test_multiple_tasks_from_config():
    cfg = {"tasks": [{"name": f"task_{i}", "type": "VoidTask"} for i in range(3)]}
    cfg = AvalancheConfig(OmegaConf.create(cfg), from_cli=False)
    ti = TaskIterator(cfg, None)
    assert len(ti.tasks) == 3
    assert sum([1 for t in ti.tasks if isinstance(t, VoidTask)]) == len(ti.tasks)
    task, changed = ti.get_task(0, 1)
    assert isinstance(task, VoidTask) and not changed


@pytest.mark.parametrize(
    "change_every, prop",
    itertools.product([1, 3, 6], ["max_task_repeat_episodes", "max_task_repeat_steps"]),
)
def test_change_task_every_ep(change_every: int, prop: str):
    cfg = {"tasks": [{"name": f"task_{i}", "type": "VoidTask"} for i in range(3)]}
    cfg.update({"task_iterator": {prop: change_every}})
    cfg = AvalancheConfig(OmegaConf.create(cfg), from_cli=False)
    ti = TaskIterator(cfg, None)

    for ep in range(13):
        prev_id = ti._active_task_idx
        prev_task = ti.tasks[ti._active_task_idx]
        task, changed = ti.get_task(ep, ep)
        assert isinstance(task, VoidTask)
        if ep > 0 and ep % change_every == 0:
            assert changed and ti._active_task_idx != prev_id and task != prev_task
        else:
            assert not changed and ti._active_task_idx == prev_id and task == prev_task


@pytest.mark.parametrize(
    "change_every, prop",
    itertools.product([1, 3], ["max_task_repeat_episodes", "max_task_repeat_steps"]),
)
def test_change_task_behavior(change_every: int, prop: str):
    cfg = {"tasks": [{"name": f"task_{i}", "type": "VoidTask"} for i in range(3)]}
    cfg.update(
        {
            "task_iterator": {
                "task_change_behavior": TASK_CHANGE_BEHAVIOR.NON_FIXED,
                "task_change_timesteps_low": 1,
                "task_change_timesteps_high": 5,
                # no changes happend if you dont set either `max_task_repeat_steps` 
                # or `max_task_repeat_episodes`
                prop: change_every
            }
        }
    )
    cfg = AvalancheConfig(OmegaConf.create(cfg), from_cli=False)
    ti = TaskIterator(cfg, None)

    for ep in range(13):
        prev_id = ti._active_task_idx
        change_every = ti._next_task_change_timestep
        prev_task = ti.tasks[ti._active_task_idx]
        task, changed = ti.get_task(ep, ep)
        assert isinstance(task, VoidTask)
        if ep > 0 and ep % change_every == 0:
            assert changed and ti._active_task_idx != prev_id and task != prev_task
        else:
            assert not changed and ti._active_task_idx == prev_id and task == prev_task