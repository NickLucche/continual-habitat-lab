from continual_habitat_lab.config import *
import pytest
import habitat_sim
from omegaconf import OmegaConf
from dataclasses import dataclass


def test_default_config_instance():
    base = ContinualHabitatLabConfig(from_cli=False)

    @dataclass
    class BaseConfig:
        experiment_name: str = "MyAvalanceExperiment"
        # tasks is a special dataclass containing all init arguments defined when creating a task
        tasks: List[base.DynamicTaskClass] = field(default_factory=lambda: [])
        task_iterator: TaskIteratorConfig = TaskIteratorConfig()
        scene: SceneConfig = SceneConfig()
        simulator: SimulatorConfig = SimulatorConfig()
        agent: AgentConfig = AgentConfig()

    base_config = OmegaConf.structured(BaseConfig)
    return base._config == base_config


def create_from_dict():
    cfg = {
        "tasks": [{"type": "VoidTask", "max_steps": 1}],
        "agent": {"sensor_specifications": [{"type": "RGB", "resolution": [128, 128]}]},
        "task_iterator": dict(max_task_repeat_episodes=1),
        "scene": {"scene_path": "/my/scene/path"},
    }
    config = ContinualHabitatLabConfig(cfg, from_cli=False)

    for k in cfg:
        assert getattr(config, k, None) is not None


def test_add_task():
    pass


def test_add_habitat_sensor_spec():
    spec = habitat_sim.SensorSpec()
    spec.uuid = "semantic"  # used for getting obs key
    spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    spec.resolution = [128, 128]
    spec.position = [0.0, 1.5, 0.0]
    config = ContinualHabitatLabConfig()
    prev_len = len(config.agent.sensor_specifications)
    config.agent.sensor_specifications.append(SensorConfig.from_sensor_spec(spec))
    assert len(config.agent.sensor_specifications) == (prev_len + 1)
    assert Sensors.SEMANTIC in [s.type for s in config.agent.sensor_specifications]

