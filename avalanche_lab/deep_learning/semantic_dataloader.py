from omegaconf import OmegaConf
import torch
import os

# remove info logging
os.environ["GLOG_minloglevel"] = "3"
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import habitat_sim
from typing import Tuple
from dataclasses import dataclass
from avalanche_lab.registry import registry
from avalanche_lab.tasks.tasks import ObjectNav, Difficulty, NavigationGoal
from avalanche_lab.config import AvalancheConfig
from avalanche_lab.env import AvalancheEnv
from avalanche_lab.tasks.navigation import generate_pointnav_episode
import random

MAX_RETRIES = 30

# custom task for exploring dataset without rewards/goal_test
@registry.register_task
class SceneExplorer(ObjectNav):
    def __init__(self, sim: habitat_sim.Simulator, *args, **kwargs) -> None:
        super().__init__(sim, *args, **kwargs)

    def goal_test(self, obs) -> bool:
        return False

    def reward_function(self, prev_obs, curr_obs, action):
        return 0.0

    def _generate_random_path(self, n: int):
        goals = []
        # generate random path from sensible starting points
        for i in range(n):
            # if not sim.pathfinder.is_loaded:
            pathf = self.sim.pathfinder
            source = pathf.get_random_navigable_point()
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            n_steps = random.randint(5, 50)
            goals.append(
                NavigationGoal(
                    source,
                    source_rotation,
                    None,
                    np.random.choice(
                        ["move_forward", "turn_right", "turn_left"], n_steps
                    ).tolist()
                    + [None],
                    None,
                    _num_iterations_to_find=None,
                )
            )
        return goals

    def _generate_goal(self):
        if not len(self.goals):
            # self.goals = generate_pointnav_episode(
            #     self.sim,
            #     number_of_episodes=self.n_episodes,
            #     geodesic_to_euclid_starting_ratio=self.difficulty.value,
            #     geodesic_to_euclid_min_ratio=1.0,
            # )
            self.goals = self._generate_random_path(self.n_episodes)
            if self.goals is None:
                raise Exception("Can't generate new goal")

        self.goal = self.goals.pop()
        if self.goal.shortest_path is None:
            raise Exception("")


explorer_config = {
    "tasks": [
        {
            "type": "SceneExplorer",
            "name": "DatasetExplorer",
            "difficulty": Difficulty.TRIVIAL,
        }
    ],
    "agent": {
        "action_space": {
            # you only need to specify the variable to pass to your action parameters
            "move_forward": {"amount": 1.0},
            "turn_right": {"amount": 45.0},
            "turn_left": {"amount": 45.0},
        },
        "sensor_specifications": [
            {"type": "RGB"},
            {"type": "SEMANTIC"},
            {"type": "DEPTH"},
        ],
    },
    "scene": {
        # 'scene_path': '/home/nick/datasets/habitat/gibson/gibson/Cokeville.glb'
        "dataset_paths": ["/home/nick/datasets/habitat/gibson/gibson/"]
        # "dataset_paths": ["/home/nick/datasets/habitat/scene_dataset/mp3d/v1/tasks/mp3d/"]
    },
}

# TODO: gather paths and assign scene id to them without invalidating em by subclassing task
# TODO: CUDA habitat sim to get observations directly on gpu
@dataclass
class VisualExplorationDataset(IterableDataset):
    r"""
    This class abstracts the exploration of a dataset intended as a set of `habitat_sim` 
    navigable scenes as a Pytorch dataset, to be used for learning from images with 
    semantic/depth groundtruth.
    To provide data, we're going to sample different points per scene and have and greedy
    agent traverse the scene gathering observations as it goes.
    """
    config: AvalancheConfig = AvalancheConfig(OmegaConf.create(explorer_config))
    dataset_size: int = int(1e5)
    img_resolution: Tuple[int, int] = (128, 128)
    semantic: bool = True
    depth: bool = False
    paths_per_scene: int = 10

    def __post_init__(self):
        # instatiate an env using ObjectNav task to explore scenes and 'modded' configs
        # iterable dataset workers are replicated on each worker therefore we'll handle multiple envs
        # self.config = OmegaConf.merge(self.config, )
        self.config.scene.max_scene_repeat_episodes = self.paths_per_scene
        self.config.tasks[0].pre_compute_episodes = self.paths_per_scene
        if not self.semantic:
            self.config.agent.sensor_specifications.pop(1)
            if not self.depth:
                self.config.agent.sensor_specifications.pop(1)
        elif not self.depth:
            self.config.agent.sensor_specifications.pop(2)

        for spec in self.config.agent.sensor_specifications:
            spec.resolution = self.img_resolution

        self.step = 0
        print(OmegaConf.to_yaml(self.config._config))
        self.env = AvalancheEnv(self.config)
        self.env.reset()
        # to explore scenes
        # self._pathfinder = self.env.sim.pathfinder

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()

        return self

    def __next__(self):
        action = self.env.current_task.goal.shortest_path[self.step]
        print('len of path', len(self.env.current_task.goal.shortest_path))
        self.step += 1
        obs, _, _, _ = self.env.step(action)
        # this is the last step, reset env/get new path
        if self.env.current_task.goal.shortest_path[self.step] is None:
            self.step = 0
            self.env.reset()
        return obs

    def __exit__(self):
        self.close()

    def close(self):
        self.env.close()


# TODO: sequential sample correlation might hurt learning?

if __name__ == "__main__":
    import cv2
    # needs to have this file to load semantic annotations
    # Loading Semantic Stage mesh : ../mp3d/v1/tasks/mp3d/ZMojNkEp431/ZMojNkEp431_semantic.ply

    dataset = VisualExplorationDataset(paths_per_scene=1, img_resolution=(512, 512))

    for i, obs in enumerate(dataset):
        print("current scene", dataset.env.current_scene)
        print("num obs", i)
        rgb = obs["rgb"].astype(np.uint8)
        print(rgb.shape, rgb.max(), rgb.min(), rgb.dtype)
        print(rgb.shape)
        cv2.imshow("rgb", rgb)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

