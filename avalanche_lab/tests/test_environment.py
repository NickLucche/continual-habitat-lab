from avalanche_lab.tasks.tasks import VoidTask
from avalanche_lab.config import AvalancheConfig
from avalanche_lab.env import AvalancheEnv
import pytest
import numpy as np
import matplotlib.pyplot as plt
from avalanche_lab.utils import suppress_habitat_logging

suppress_habitat_logging()

def test_empty_config_env():
    with AvalancheEnv(AvalancheConfig(from_cli=False)) as env:
        assert isinstance(env.current_task, VoidTask)


@pytest.mark.parametrize("n_task", [1, 3, 5])
def test_env(n_task: int):
    res = [80, 80]
    cfg = {
        "tasks": [{"type": "VoidTask", "max_steps": 1} for i in range(n_task)],
        "agent": {"sensor_specifications": [{"type": "RGB", 'resolution': res}]},
    }
    config = AvalancheConfig(cfg, from_cli=False)
    with AvalancheEnv(config) as env:
        task_idx = 0
        # action_names = list(
        # env.scene_manager.habitat_config().agents[0].action_space.keys()
        # )
        for _ in range(1):
            env.reset()
            assert (
                env.task_iterator._active_task_idx == task_idx
            ), "Task should change at each new episode"

            task_idx += 1
            # execute random action
            action = env.action_space.sample()
            print("action", action)
            observation, _, done, _ = env.step(action)
            assert done
            rgb = observation["rgb"]
            assert rgb.shape[0] == res[0] and rgb.shape[1] == res[1] 
            assert 'depth' not in observation
            assert 'semantic' not in observation


def test_env_with_custom_tasks():
    pass
