# Continual Habitat Lab
---
Continual Habitat Lab is an alternative (yet inspired) to Habitat-Lab, providing a high-level abstraction over Habitat-Sim with a specific focus on Continual Learning and Reinforcement Learning. CHL greatly simplifies and 'shrinks' the abstraction layers by reducing explicitely to the RL agent-environment interaction schema and natively adopting `OpenAI Gym` interface:
 - revisits the concept of `Task`, in a simple and yet closer to a formal definition way: a `Task` must define a reward function `R(s, s', a)->r`, a goal test function `G(s)->done` and an action space `A` (actions mapping). A task can then freely access the `Habitat` simulator and has access to a special callbacks to act upon it. Furthermore, all `__init__` args that make your task tunable are by default available as configuration fields, so you can run multiple experiments by simply changing parameters on a `.yaml` file. 
 - the environment is by default an `OpenAI Gym` environment and groups together these components: a `habitat `simulator, a `TaskIterator` and a `SceneManager`. In a `env.step()` call the `habitat` simulator provides the observation while the current task provides reward + termination flag. 
 - A `TaskIterator` allows to iterate over a collection of tasks with configurable parameters in terms of number of steps/episodes each one should be run for, providing access to the current 'active' task. This 'stream of tasks' concept is natively supported as it is at the core of Continual Learning definition.   
 - A `SceneManager` allows to define how to switch and iterate between scenes during environment interaction (e.g. under which conditions the scene should change..) with the goal of making this process super easy and again natively supported, as this feature is also of great importance for CL. You can go over scenes contained in a dataset or in a collection of dataset by simply moving them to a common folder, all by changing a couple of lines in your `.yaml` config file.
 - features a more powerful configuration system entirely based on `OmegaConf`, which integrates all configurations options from `habitat sim` for instatiating the simulator so you don't have to re-learn new terminologies for the same stuff. 
 - observation space, what the __agent__ can perceive, is implicitly defined in configuration by setting the available sensors. 
 - drops the explicit need of `Datasets` you have in `Habitat-Lab`: data should be generated through environment interactions (enforcing RL view) and initial states definition should be an option entirely up to the developer, achievable in many ways (e.g. by subclassing the simulator or defining a new `Task`).
 - new actions can be defined leveraging the standard `habitat sim` registry. 

This figure shows a bit what it's written above in great lenght:



## Quick Example
```python
from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
import random

# Load config from yaml file 
config = ContinualHabitatLabConfig.from_yaml("example_config.yaml")
# ..or use default one
config = ContinualHabitatLabConfig()
# ..or even create it programatically
cfg = {
    "tasks": [{"type": "VoidTask", "max_steps": 10, 'name': 'QuickExampleTask'}],
    "agent": {"sensor_specifications": [{"type": "RGBA", "resolution": [128, 128]}]},
    "scene": {"scene_path": "./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"},
}
config = ContinualHabitatLabConfig(cfg, from_cli=False)
config.simulator.random_seed = 7

# check out good ol' habitat-sim configuration to see registered actions
action_names = list(config.habitat_sim_config.agents[0].action_space.keys())
print(action_names)
n_episodes = 2
with ContinualHabitatEnv(config) as env:
    print("\n"*10 + "*"*50 + '\n'+"*"*50)
    print("Current scene:", env.current_scene)
    print("Available tasks:", [t.name for t in env.tasks])
    for _ in range(n_episodes):
        obs = env.reset()
        print("Current task:", env.current_task)
        print("Initial position", env.agent_position)

        while not env.done:
            # execute random action
            action = env.action_space.sample()
            # get the action space mapping as defined by the task
            action_mapping = env.current_task._action_space_map
            print("Action", action_mapping[action])
            
            obs, reward, done, _ = env.step(action)
            print("Current position/orientation", env.agent_position, env.agent_rotation)

        print("Episode terminated!")
```

# Installation

Head over to https://github.com/facebookresearch/habitat-sim#installation for the latest instructions on how to install `habitat-sim`, containing the actual simulator engine. At the time of writing, this is as easy as doing `conda install habitat-sim -c conda-forge -c aihabitat` (desktop pc).
Follow the instructions to get some samples and test out the installation.

To try out a simple example, try running `p example.py -i` (interactive) after changing the `scene_path` and `object_asset` filepaths in `example_config.yaml`.

