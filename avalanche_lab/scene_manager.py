# handle scene switch logic given a list of paths of scene dataset
from avalanche_lab.config import AvalancheConfig
from typing import Iterator, List, Dict
from pathlib import Path, PosixPath
import random
from collections import OrderedDict
from itertools import cycle
import logging


class SceneManager:
    _config: AvalancheConfig
    _allowed_scene_extensions = ["glb", "ply", "gltf", "obj"]
    _scenes_by_dataset: Dict[str, List[PosixPath]]
    _current_dataset: str
    _current_scene: str = None
    # for cycling though datasets
    _dataset_iterator: Iterator
    # for iterating over scenes in a dataset
    _scene_iterator: Iterator

    def __init__(self, config: AvalancheConfig) -> None:
        self._config = config
        self._cycle_datasets = self._config.scene.cycle_datasets
        self._sample_random_scene = self._config.scene.sample_random_scene

        self._init_scene_manager()

    def _init_scene_manager(self):
        if not self._config.scene.dataset_paths or not len(
            self._config.scene.dataset_paths
        ):
            raise Exception(
                "No scene directory specified in configuration. Make sure to select a scene before starting the environment!"
            )
        self._scenes_by_dataset = OrderedDict()
        # check that every (super) directory contains at least one valid mesh
        for dirpath in self._config.scene.dataset_paths:
            paths = []
            # I did consider using generators directly but it doesnt seem worth
            for ext in self._allowed_scene_extensions:
                paths.extend(Path(dirpath).rglob(f"*.{ext}"))

            if not len(paths):
                raise Exception(
                    "No scene with accepted format ({}) found at {}. Can't initialize Simulator.".format(
                        self._allowed_scene_extensions, dirpath
                    )
                )
            # TODO:
            # if self._config.split_subdirectories

            self._scenes_by_dataset[dirpath] = paths

        # initialize scene manager state
        self._dataset_iterator = cycle(self._scenes_by_dataset)
        self._current_dataset = next(self._dataset_iterator)
        self._scene_iterator = iter(self._scenes_by_dataset[self._current_dataset])
        if self._cycle_datasets:
            self._current_scene = str(next(self._scene_iterator))
        elif self._sample_random_scene:
            self._current_scene = str(self._sample_scene_from_dataset())
        else:
            raise NotImplementedError(
                "Make sure at least one flag among `cycle_datasets` and `sample_random_scene` is set"
            )

    def get_scene(self, episode_counter: int):
        """
        Return scene to use as specified by the configuration policy.

        Args:
            episode_counter (int): [description]
        """

        # check whether we need to change scene
        changed = self._change_scene(episode_counter)
        if changed:
            if self._cycle_datasets:
                scene = self._iterate_dataset()
                # we iterated over all scenes in this dataset
                if scene is None:
                    self._current_dataset = next(self._dataset_iterator)
                    self._scene_iterator = iter(
                        self._scenes_by_dataset[self._current_dataset]
                    )
                    scene = self._iterate_dataset()
            elif self._sample_random_scene:
                # get a random scene from a random dataset
                dataset = random.choice(list(self._scenes_by_dataset.keys()))
                self._current_dataset = dataset
                scene = self._sample_scene_from_dataset()
            logging.info("Scene changed to {}.".format(scene))
            self._current_scene = str(scene)
        return self._current_scene, changed

    def _sample_scene_from_dataset(self, scenes_path: str = None) -> PosixPath:
        if scenes_path is None:
            scenes_path = self._current_dataset
        return random.choice(self._scenes_by_dataset[scenes_path])

    def _iterate_dataset(self, scene_iter: Iterator = None) -> PosixPath:
        if scene_iter is None:
            scene_iter = self._scene_iterator
        return next(scene_iter, None)

    def _change_scene(self, ep_counter: int):
        max_ep = self._config.scene.max_scene_repeat_episodes
        if max_ep > 0 and ep_counter > 0 and ep_counter % max_ep == 0:
            return True
        return False

