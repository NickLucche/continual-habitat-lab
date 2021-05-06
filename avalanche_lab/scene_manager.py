# handle scene switch logic given a list of paths of scene dataset 
from avalanche_lab.config import AvalancheConfig
from habitat_sim import SimulatorConfiguration, CameraSensorSpec, Configuration, SensorType
from habitat_sim.agent import AgentConfiguration

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return Configuration(sim_cfg, [agent_cfg])


sim_settings = {
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}

class SceneManager:
    _config: AvalancheConfig
    def __init__(self, config: AvalancheConfig) -> None:
        self._config = config

    def habitat_config(self):
        # TODO: pick scene from dataset
        sim_settings['scene'] = self._config.scene_handler.scenes_paths[0]
        return make_simple_cfg(sim_settings)


