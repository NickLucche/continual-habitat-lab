from continual_habitat_lab.config import ContinualHabitatLabConfig, SceneConfig
import pytest
from continual_habitat_lab.scene_manager import SceneManager
import os


def skip_with_no_datasets(cfg: ContinualHabitatLabConfig):
    for d in cfg.scene.dataset_paths:
        if not os.path.exists(d):
            pytest.skip(
                f"Directory {d} not found, set dataset directory accordingly before running tests"
            )


def test_no_dataset():
    config = ContinualHabitatLabConfig(from_cli=False)
    config.scene.dataset_paths = []
    with pytest.raises(Exception):
        SceneManager(config)


def test_scene_change_every_ep():
    config = ContinualHabitatLabConfig(from_cli=False)
    config.scene.max_scene_repeat_episodes = 1
    skip_with_no_datasets(config)
    sm = SceneManager(config)
    # simulate simulator steps by increasing episode counter passed
    for ep in range(10):
        _, changed = sm.get_scene(ep)
        if ep > 0:
            assert changed == True
        else:
            assert changed == False

def test_scene_change_every_step():
    config = ContinualHabitatLabConfig(from_cli=False)
    config.scene.max_scene_repeat_episodes = -1
    config.scene.max_scene_repeat_steps = 1
    skip_with_no_datasets(config)
    sm = SceneManager(config)
    # simulate simulator steps 
    for step in range(10):
        _, changed = sm.get_scene(-1, step)
        if step > 0:
            assert changed == True
        else:
            assert changed == False


def test_gibson():
    config = ContinualHabitatLabConfig(from_cli=False)
    config.scene.dataset_paths = ["data/gibson/"]
    config.scene.max_scene_repeat_episodes = 1
    skip_with_no_datasets(config)
    sm = SceneManager(config)
    # simulate simulator steps by increasing episode counter passed
    for ep in range(10):
        _, changed = sm.get_scene(ep)
        if ep > 0:
            assert changed == True
        else:
            assert changed == False