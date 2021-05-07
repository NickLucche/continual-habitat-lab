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
@dataclass
class SimulatorConfig:
    # TODO: instatiate from SimConfig() class
    # scene_id: str = '' this is ignored anyway
    default_agent_id: int = 0


@dataclass
class SceneConfig:
    # dataset paths, recursively finds scenes files inside each directory 
    dataset_paths: List[str] = field(
        default_factory=lambda: [
            "data/scene_datasets/habitat-test-scenes/"
        ]
    )
    split_dataset_subdirectories: bool = False
    max_scene_repeat_episodes: int = -1
    # cycle datasets, can't move to other dataset before all scenes have been selected 
    cycle_datasets: bool = False
    # sample next scene from any specified dataset
    sample_random_scene: bool = True
    # TODO: move to sim?
    change_lighting: bool = False


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
    scene: SceneConfig = SceneConfig()
    simulator:SimulatorConfig = SimulatorConfig()


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

    @staticmethod
    def habitat_sim_config(config: 'AvalancheConfig', scene: str):
        import habitat_sim
        sim_cfg = habitat_sim.SimulatorConfiguration()
        for k, v in config.simulator.items():
            setattr(sim_cfg, k, v)

        sim_cfg.scene_id = scene
        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [512, 512]
        rgb_sensor_spec.position = [0.0, 2.0, 0.0]

        agent_cfg.sensor_specifications = [rgb_sensor_spec]

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def __eq__(self, o: object) -> bool:
        return self.config == o.config
