import habitat_sim
import numpy as np
# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor" # used for getting obs key 
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# This is the scene we are going to load.
# we support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
test_scene = "/home/nick/datasets/habitat/gibson/gibson/Cokeville.glb"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}
cfg = make_simple_cfg(sim_settings)
# create simulator
with habitat_sim.Simulator(cfg) as sim:
    # init agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    total_frames = 0
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    max_frames = 10
    import random
    import matplotlib.pyplot as plt
    while total_frames < max_frames:
        action = random.choice(action_names)
        print("action", action)
        observations = sim.step(action)
        obs = observations["color_sensor"]
        # obs = observations["semantic_sensor"]
        # depth = observations["depth_sensor"]
        print(obs.shape)
        plt.imshow(obs)
        plt.show()
        total_frames += 1