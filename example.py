from omegaconf.omegaconf import OmegaConf
from avalanche_lab.config import AvalancheConfig
from avalanche_lab.env import Env
import pytest
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from argparse import ArgumentParser
import random
import habitat_sim

sys.path.insert(0, "/home/nick/uni/thesis/avalanche-lab/")

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"


# def rgb2bgr(image):
    # return image[..., ::-1]


def visualize(flag: bool, obs):
    if flag:
        cv2.imshow("RGB", obs["rgb"])


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "-i", "--interactive", help="Run demo interactively", action="store_true",
    )
    args = args.parse_args()
    n_episodes = 3


    # config = AvalancheConfig(from_cli=False)
    config = AvalancheConfig.from_yaml('example_config.yaml')
    config.scene.max_scene_repeat_episodes = -1


    print("Simulator configuration:\n", OmegaConf.to_yaml(config._config) )
    with Env(config) as env:
        task_idx = 0
        action_names = list(
            config.habitat_sim_config.agents[0].action_space.keys()
        )
        print("Available actions", action_names)
        print(config.habitat_sim_config.agents[0].action_space)
        print("Current scene:", env.scene_manager._current_scene)
        for _ in range(n_episodes):
            end = False
            obs = env.reset()
            visualize(args.interactive, obs)
            # assert env._curr_task_idx == task_idx, "Task should change at each new episode"
            task_idx = (task_idx + 1) % 2
            # TODO: while not env.episode_over:
            for i in range(3):
                if args.interactive:
                    keystroke = cv2.waitKey(0)
                    # ord gets unicode from one-char string
                    if keystroke == ord(FORWARD_KEY):
                        action = "move_forward"
                    elif keystroke == ord(LEFT_KEY):
                        action = "turn_left"
                    elif keystroke == ord(RIGHT_KEY):
                        action = "turn_right"
                    elif keystroke == ord(FINISH):
                        action = "stop"
                    elif keystroke == ord("q") or keystroke == 127:
                        print("Closing..")
                        end = True
                        break
                    else:
                        print("INVALID KEY")
                        continue
                    # TODO:
                    # if action not in env.task.actions:
                    #     print("Invalid action!")
                    #     continue
                else:
                    # execute random action
                    # TODO: action = env.action_space.sample()
                    action = random.choice(action_names)
                # env.step(env.action_space.sample())
                print("action", action)
                obs = env.step(action)
                visualize(args.interactive, obs)

            if end:
                break
    cv2.destroyAllWindows()
