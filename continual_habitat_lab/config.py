from omegaconf import OmegaConf, MISSING
import omegaconf
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import asdict, dataclass, field, make_dataclass
import habitat_sim
import enum
import numpy as np


class TASK_CHANGE_BEHAVIOR(enum.IntEnum):
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


def make_dynamic_dataclass(class_name: str, base_class):
    from continual_habitat_lab.registry import registry
    import inspect

    # TODO: issue with arguments with same name and different types.
    tasks_classes = registry.get_all("task")
    print("tasks classes", tasks_classes)
    signs = map(lambda C: inspect.getfullargspec(C.__init__), tasks_classes.values())
    ignored_arguments = ["self", "sim"]

    fields = []
    for s in signs:
        arg_counter = 0
        for arg in s.args:
            if arg not in ignored_arguments:
                # if no default argument value was given set field to None
                if s.defaults is None:
                    fields.append((arg, Optional[s.annotations[arg]], None))
                else:
                    fields.append((arg, s.annotations[arg], s.defaults[arg_counter]))
                arg_counter += 1
    print("Tasks fields", fields)
    return make_dataclass(class_name, fields=fields, bases=(base_class,))


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


class Sensors(enum.IntEnum):
    RGBA = 0
    DEPTH = 1
    SEMANTIC = 2
    GPS = 3


@dataclass
class SensorConfig:
    # way to quickly setup sensors with good defaults
    type: Sensors = Sensors.RGBA
    uuid: str = ""
    # sensor_type: habitat_sim.SensorType = habitat_sim.SensorType.COLOR
    # sensor_subtype: habitat_sim.SensorSubType = habitat_sim.SensorSubType.PINHOLE
    parameters: Optional[Dict[str, str]] = None
    resolution: List[int] = field(default_factory=lambda: [128, 128])
    position: List[float] = field(default_factory=lambda: [0.0, 1.5, 0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @staticmethod
    def from_sensor_spec(spec: habitat_sim.SensorSpec):
        """ to modify config programmatically using habitat sim way like
        ```
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "rgba"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            ...
            config.agent.sensor_specifications.append(SensorConfig.from_sensor_spec(rgb_sensor_spec))
        ```

        Args:
            spec (habitat_sim.SensorSpec): habitat sim sensor specification object.

        Raises:
            NotImplementedError: If the sensor hasn't been added yet.
        Returns:
            An instance of SensorConfig.
        """
        stype = None
        if spec.sensor_type == habitat_sim.SensorType.COLOR:
            stype = Sensors.RGBA
        elif spec.sensor_type == habitat_sim.SensorType.SEMANTIC:
            stype = Sensors.SEMANTIC
        elif spec.sensor_type == habitat_sim.SensorType.DEPTH:
            stype = Sensors.DEPTH
        else:
            raise NotImplementedError("Unknown sensor")
        params = {}
        for k in SensorConfig.__annotations__.keys():
            if k != "type":
                v = getattr(spec, k, None)
                v = v.tolist() if isinstance(v, np.ndarray) else v
                params[k] = v

        return SensorConfig(type=stype, **params)


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
    sensor_specifications: List[SensorConfig] = field(
        default_factory=lambda: [SensorConfig()]
    )

    def __post_init__(self):
        # TODO: how to disable default actions from config?
        for k in asdict(self):
            if k not in ["action_space", "sensor_specifications"]:
                setattr(self, k, _default_agent_cfg[k])


@dataclass
class SceneConfig:
    # dataset paths, recursively finds scenes files inside each directory
    dataset_paths: List[str] = field(
        default_factory=lambda: ["data/scene_datasets/habitat-test-scenes/"]
    )
    # if set inhibits SceneManager behavior and only returns this single scene specified
    scene_path: Optional[str] = None
    split_dataset_subdirectories: bool = False
    max_scene_repeat_episodes: int = -1
    max_scene_repeat_steps: int = -1
    # cycle datasets, can't move to other dataset before all scenes have been selected
    cycle_datasets: bool = False
    # sample next scene from any specified dataset
    sample_random_scene: bool = True


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


class ContinualHabitatLabConfig(object):

    _config: OmegaConf
    _sim_cfg: habitat_sim.SimulatorConfiguration
    _agent_cfgs: List[habitat_sim.AgentConfiguration]

    def __init__(
        self,
        config: Union[OmegaConf, Dict[str, Any]] = OmegaConf.create(),
        from_cli: bool = False,
    ) -> None:
        # workaround to get dynamic `Tasks` dataclass to be evaluated at config
        # instatiation time rather than import time (solve import order bug)
        self.DynamicTaskClass = make_dynamic_dataclass("DynamicTaskClass", TaskConfig)

        @dataclass
        class BaseConfig:
            experiment_name: str = "MyAvalanceExperiment"
            # tasks is a special dataclass containing all init arguments defined when creating a task
            tasks: List[self.DynamicTaskClass] = field(default_factory=lambda: [])
            task_iterator: TaskIteratorConfig = TaskIteratorConfig()
            scene: SceneConfig = SceneConfig()
            simulator: SimulatorConfig = SimulatorConfig()
            agent: AgentConfig = AgentConfig()

        base_config = OmegaConf.structured(BaseConfig)
        if type(config) is dict:
            config = OmegaConf.create(config)

        if from_cli:
            config = OmegaConf.merge(config, OmegaConf.from_cli())
        self._config = OmegaConf.merge(base_config, config)
        self._override_habitat_sim_config()

    @staticmethod
    def from_yaml(filepath: str, from_cli=False) -> "ContinualHabitatLabConfig":
        return ContinualHabitatLabConfig(OmegaConf.load(filepath), from_cli=from_cli)

    @property
    def habitat_sim_config(self):
        # When we edit simulator or agent habitat sim config, habitat config
        # should be updated as well, therefore we re-instantiate it
        # self._override_habitat_sim_config()
        return habitat_sim.Configuration(self._sim_cfg, self._agent_cfgs)

    def refresh_config(self):
        return self._override_habitat_sim_config()

    def make_habitat_sim_config(self, scene: str):
        # we need to re-instatiate the habitat-sim configuration or the simulator
        # won't recognize differences with previous object
        self._override_habitat_sim_config()
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
                from continual_habitat_lab.registry import registry

                # read registered action parameters and instatiate those specified in config
                print("all", v)
                for (
                    action_key,
                    action_param_class,
                ) in registry.get_all_action_params().items():
                    # for action_key, params in v.items():
                    # TODO: How to merge with default values?
                    params = v.get(action_key, {})
                    print(
                        "key",
                        action_key,
                        "val",
                        params,
                        "from registry",
                        action_param_class,
                    )
                    # default value merging carried out in `ActionParameters` subclasses
                    agent_cfg.action_space[action_key] = habitat_sim.ActionSpec(
                        action_key, action_param_class(**params)
                    )
            elif k == "sensor_specifications":
                sensors = list(map(lambda sc: self._create_sensor(sc), v))
                print("SENSORS", sensors)
                agent_cfg.sensor_specifications = sensors
            else:
                setattr(agent_cfg, k, v)

        self._agent_cfgs = [agent_cfg]

    def _create_sensor(self, sensor_spec: SensorConfig):
        def _create_camera(
            spec_dict: omegaconf.dictconfig.DictConfig,
            sensor_type: habitat_sim.SensorType,
        ):
            # spec_dict = asdict(spec)
            type_ = spec_dict["type"]
            if spec_dict["uuid"].strip() == "":
                spec_dict["uuid"] = type_.name.lower()
            camera = habitat_sim.CameraSensorSpec()
            # set it by default
            camera.sensor_type = sensor_type
            # camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            for k, v in spec_dict.items():
                if k != "type":
                    setattr(camera, k, v)
            return camera

        if sensor_spec.type == Sensors.RGBA:
            return _create_camera(sensor_spec, habitat_sim.SensorType.COLOR)
        elif sensor_spec.type == Sensors.SEMANTIC:
            return _create_camera(sensor_spec, habitat_sim.SensorType.SEMANTIC)
        elif sensor_spec.type == Sensors.DEPTH:
            return _create_camera(sensor_spec, habitat_sim.SensorType.DEPTH)

