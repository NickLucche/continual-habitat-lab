from omegaconf import OmegaConf
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass, field
import enum


class TASK_CHANGE_BEHAVIOR(enum.IntEnum):
    # TODO: works with lowercase too?
    FIXED = 0
    NON_FIXED = 1


class TASK_SAMPLING(enum.IntEnum):
    SEQUENTIAL = 0
    RANDOM = 1


# default configuration here
# @dataclass
# class SimulatorConfig:


@dataclass
class SceneConfig:
    # TODO: dataset paths
    scenes_paths: List[str] = field(
        default_factory=lambda: [
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        ]
    )


@dataclass
class TaskConfig:
    name: str
    type: str


@dataclass
class TaskIteratorConfig:
    task_sampling: TASK_SAMPLING = TASK_SAMPLING.SEQUENTIAL
    task_change_behavior: TASK_CHANGE_BEHAVIOR = TASK_CHANGE_BEHAVIOR.FIXED
    max_task_repeat_steps: Optional[int] = None
    max_task_repeat_episodes: Optional[int] = None
    start_from_random_task: bool = False
    # used with TASK_CHANGE_BEHAVIOR.NON_FIXED to define range of task change in steps/episodes
    task_change_timesteps_low: int = -1
    task_change_timesteps_high: int = -1


@dataclass
class BaseConfig:
    experiment_name: str = "MyAvalanceExperiment"
    tasks: List[TaskConfig] = field(default_factory=lambda: [])
    task_iterator: TaskIteratorConfig = TaskIteratorConfig()
    scene_handler: SceneConfig = SceneConfig()


base_config = OmegaConf.structured(BaseConfig)


class AvalancheConfig:
    config: OmegaConf

    def __init__(
        self, config: OmegaConf = OmegaConf.create(), from_cli: bool = True
    ) -> None:
        if from_cli:
            config = OmegaConf.merge(config, OmegaConf.from_cli())
        self.config = OmegaConf.merge(base_config, config)

    @staticmethod
    def from_yaml(filepath: str) -> "AvalancheConfig":
        return AvalancheConfig(OmegaConf.load(filepath))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def __eq__(self, o: object) -> bool:
        return self.config == o.config
