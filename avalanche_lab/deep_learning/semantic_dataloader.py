from avalanche_lab.config import AvalancheConfig
from avalanche_lab.env import AvalancheEnv
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import habitat_sim
from typing import Tuple
from dataclasses import dataclass
from avalanche_lab.registry import registry
from avalanche_lab.tasks.tasks import ObjectNav, Difficulty

MAX_RETRIES = 30

# custom task for exploring dataset without rewards/goal_test
@registry.register_task
class SceneExplorer(ObjectNav):
    def __init__(self, sim: habitat_sim.Simulator, *args, **kwargs) -> None:
        super().__init__(sim, *args, **kwargs)

    def goal_test(self, obs) -> bool:
        return False

    def reward_function(self, prev_obs, curr_obs, action):
        return 0.


explorer_config = {
    "tasks": [{"type": "SceneExplorer", "name": "DatasetExplorer", 'difficulty': Difficulty.TRIVIAL, }],
    "agent": {
        "action_space": {
            "move_forward": habitat_sim.ActionSpec(
                "move_forward", habitat_sim.ActuationSpec(amount=1.0)
            ),
            "turn_left": habitat_sim.ActionSpec(
                "turn_left", habitat_sim.ActuationSpec(amount=45.0)
            ),
            "turn_right": habitat_sim.ActionSpec(
                "turn_right", habitat_sim.ActuationSpec(amount=45.0)
            ),
        }
    },
}


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
    config: AvalancheConfig = AvalancheConfig()
    dataset_size: int = int(1e5)
    obs_shape: Tuple[int, int] = (128, 128)
    semantic: bool = True
    depth: bool = False
    paths_per_scene: int = 10

    def __post__init__(self):
        # instatiate an env using ObjectNav task to explore scenes and 'modded' configs
        # iterable dataset workers are replicated on each worker therefore we'll handle multiple envs

        self.config.scene.max_scene_repeat_episodes = self.paths_per_scene
        self.config.tasks[0].pre_compute_episodes = self.paths_per_scene
        if not self.semantic:
            # TODO: add sensor
            pass

        self.env = AvalancheEnv(self.config)

        # to explore scenes
        self._pathfinder = self.env.sim.pathfinder

        self._follower = habitat_sim.nav.GreedyGeodesicFollower(
            self._pathfinder,
            self.env.sim.get_agent(0),
            0.5,
            fix_thrashing=True,
            thrashing_threshold=16,  # number of actions to stop being a rumba in a corner
        )
        self._paths = []

    def _set_random_start_pos(self):
        source = self._pathfinder.get_random_navigable_point()
        agent = self.env.sim.get_agent(0)
        state = agent.get_state()
        state.position = source
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        state.rotation = source_rotation
        state.sensor_states = {}
        agent.set_state(state, reset_sensors=True)
        self.env.sim.reset()
        return agent.get_state()

    def _generate_path(self):
        shortest_path = None
        state = self._set_random_start_pos()
        _retry = 0
        while shortest_path is None and _retry < MAX_RETRIES:
            try:
                shortest_path = self._follower.find_path(
                    self._pathfinder.get_random_navigable_point()
                )
            except habitat_sim.errors.GreedyFollowerError as e:
                _retry += 1
        return (state, shortest_path)

    def _explore(self):
        if not len(self._paths):
            self._paths = [self._generate_path() for _ in range(self.paths_per_scene)]
        state, path = self._paths.pop()
        # TODO: self.env.sim
        obs = self.env.reset()
        yield obs
        for action_key in path:
            obs = self.env.step(action_key, dt=1 / 60)
            yield obs

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        return self

    def __next__(self):
        return self._explore()

    def __exit__(self):
        self.close()

    def close(self):
        self.env.close()


# TODO: iterable dataset, although sequential sample correlation might hurt learning?
