from avalanche_lab.config import AvalancheConfig
from avalanche_lab.env import Env
import pytest
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, "/home/nick/uni/thesis/avalanche-lab/")

config = AvalancheConfig(from_cli=True)
with Env(config) as env:
    # we have two tasks, we'll only need task index 0 and 1
    task_idx = 0
    action_names = list(
        env.scene_manager.habitat_config().agents[0].action_space.keys()
    )
    for _ in range(5):
        env.reset()
        assert env._curr_task_idx == task_idx, "Task should change at each new episode"
        task_idx = (task_idx + 1) % 2
        # while not env.episode_over:
        for i in range(3):
            # execute random action
            # env.step(env.action_space.sample())
            action = np.random.choice(action_names, 1)
            print("action", action)
            observations = env.step(action)
            rgb = observations["color_sensor"]
            print(rgb.shape)
            plt.imshow(rgb)
            plt.show()
