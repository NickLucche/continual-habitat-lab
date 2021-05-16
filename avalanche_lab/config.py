from omegaconf import OmegaConf, MISSING
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import asdict, dataclass, field
import habitat_sim
import enum


class TASK_CHANGE_BEHAVIOR(enum.IntEnum):
    # TODO: works with lowercase too?
    FIXED = 0
    NON_FIXED = 1


class TASK_SAMPLING(enum.IntEnum):
    SEQUENTIAL = 0
    RANDOM = 1


# default configuration here

# expose simualator backend options as configs with proper defaults
def get_default_sim_cfg() -> Dict:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    return {
        k: getattr(sim_cfg, k)
        for k in habitat_sim.SimulatorConfiguration.__dict__.keys()
        if not k.startswith("_")
    }


def get_default_agent_cfg() -> Dict:
    agent_cfg = habitat_sim.AgentConfiguration()
    return {
        k: getattr(agent_cfg, k)
        for k in habitat_sim.AgentConfiguration.__dict__.keys()
        if not k.startswith("_")
    }


_default_sim_cg = get_default_sim_cfg()


@dataclass
class SimulatorConfig:
    scene_dataset_config_file: str = None
    scene_id: str = None
    random_seed: int = None
    default_agent_id: int = None
    default_camera_uuid: str = None
    gpu_device_id: int = None
    allow_sliding: bool = None
    create_renderer: bool = None
    frustum_culling: bool = None
    enable_physics: bool = None
    enable_gfx_replay_save: bool = None
    physics_config_file: str = None
    override_scene_light_defaults: bool = None
    scene_light_setup: str = None
    load_semantic_mesh: bool = None
    force_separate_semantic_scene_graph: bool = None
    requires_textures: bool = None

    def __post_init__(self):
        for k in asdict(self):
            setattr(self, k, _default_sim_cg[k])


_default_agent_cfg = get_default_agent_cfg()


@dataclass
class SensorConfig:
    uuid: str = None
    sensor_type: habitat_sim.SensorType = None
    sensor_subtype: habitat_sim.SensorSubType = None
    parameters: Dict[str, str] = None
    resolution: List[float] = None
    position: List[float] = None
    orientation: List[float] = None

    def from_camera_sensor_spec(self, spec: habitat_sim.CameraSensorSpec):
        # TODO:
        pass


from habitat_sim.agent.agent import _default_action_space


@dataclass
class ActionParamsConfig:
    # TODO: this way we can keep default values and override a subset through config
    amount: float
    
@dataclass
# TODO: handle multi-agent
class AgentConfig:
    action_space: Dict[str, Any] = field(
        default_factory=lambda: {
            k: {"amount": v.actuation.amount}
            for k, v in _default_action_space().items()
        }
    )
    angular_acceleration: float = None
    angular_friction: float = None
    body_type: str = None
    coefficient_of_restitution: float = None
    height: float = None
    linear_acceleration: float = None
    linear_friction: float = None
    mass: float = None
    radius: float = None
    sensor_specifications: List[SensorConfig] = field(default_factory=lambda: [])

    def __post_init__(self):
        # TODO: how to disable default actions from config?
        for k in asdict(self):
            if k != "action_space":
                setattr(self, k, _default_agent_cfg[k])


@dataclass
class SceneConfig:
    # dataset paths, recursively finds scenes files inside each directory
    dataset_paths: List[str] = field(
        default_factory=lambda: ["data/scene_datasets/habitat-test-scenes/"]
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
    name: str = ""
    type: str = MISSING


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
    simulator: SimulatorConfig = SimulatorConfig()
    agent: AgentConfig = AgentConfig()


base_config = OmegaConf.structured(BaseConfig)


class AvalancheConfig(object):
    _config: OmegaConf
    _sim_cfg: habitat_sim.SimulatorConfiguration
    _agent_cfgs: List[habitat_sim.AgentConfiguration]

    def __init__(
        self, config: OmegaConf = OmegaConf.create(), from_cli: bool = True
    ) -> None:
        if from_cli:
            config = OmegaConf.merge(config, OmegaConf.from_cli())
        self._config = OmegaConf.merge(base_config, config)
        self._override_habitat_sim_config()

    @staticmethod
    def from_yaml(filepath: str, from_cli=False) -> "AvalancheConfig":
        return AvalancheConfig(OmegaConf.load(filepath), from_cli=from_cli)

    @property
    def habitat_sim_config(self):
        # When we edit simulator or agent habitat sim config, avalanche config
        # should be updated as well, therefore we re-instantiate it
        self._override_habitat_sim_config()
        return habitat_sim.Configuration(self._sim_cfg, self._agent_cfgs)

    def make_habitat_sim_config(self, scene: str):
        self._sim_cfg.scene_id = scene
        return habitat_sim.Configuration(self._sim_cfg, self._agent_cfgs)

    def __getattr__(self, name: str) -> Any:
        # only called if no attribute match the name since we subclassed object
        return getattr(self._config, name)

    def __getitem__(self, key):
        return self._config[key]

    def __eq__(self, o: object) -> bool:
        return self._config == o._config

    def _override_habitat_sim_config(self):
        # override default backend simulator options with the ones from configuration
        self._sim_cfg = habitat_sim.SimulatorConfiguration()
        # TODO: ignore invalid keys inserted in config?
        for k, v in self._config.simulator.items():
            setattr(self._sim_cfg, k, v)
        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        for k, v in self._config.agent.items():
            if k == "action_space":
                from avalanche_lab.registry import registry

                # read registered action parameters and instatiate those specified in config
                print("all", v)
                for action_key, action_param_class in registry.get_all_action_params().items():
                # for action_key, params in v.items():
                    # TODO: How to merge with default values?
                    params = v.get(action_key, {})
                    print("key", action_key, "val", params, 'from registry', action_param_class)
                    # action_param = registry.get_action_params(action_key)
                    # default value merging carried out in `ActionParameters` subclasses
                    agent_cfg.action_space[action_key] = habitat_sim.ActionSpec(
                        action_key, action_param_class(**params)
                    )
            else:
                setattr(agent_cfg, k, v)
        # self.action_space['no_op'] = habitat_sim.ActionSpec('no_op', None)
        # FIXME: cant assign a CameraSensorSpec object to configuration
        # attach RGB visual sensor to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [512, 512]
        rgb_sensor_spec.position = [0.0, 2.0, 0.0]
        # config.agent.sensor_specifications = [rgb_sensor_spec]
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
        self._agent_cfgs = [agent_cfg]

