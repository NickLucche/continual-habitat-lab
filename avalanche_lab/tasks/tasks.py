# easily set up tasks in any scene without so much as a sweat
from typing import Dict
from habitat_sim import Simulator
import gym.spaces.dict
import gym.spaces.discrete
from gym.spaces.space import Space
import logging
import numpy as np
import enum


class Task:
    sim: Simulator
    reward_range = (-float("inf"), float("inf"))
    # action and obs space as defined by OpenAI Gym
    action_space = gym.spaces.discrete.Discrete(3)
    # observation_space: Space TODO: ignore for now
    name: str

    def __init__(self, sim: Simulator, name:str=None, *args, **kwargs) -> None:
        self.sim = sim
        actions_key = ["turn_right", "turn_left", "move_forward"]
        self._action_space_map = {str(i): actions_key[i] for i in range(3)}
        self.name = self.__class__.__name__ if name is None else name

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        raise NotImplementedError()

    def reward_function(
        self, prev_obs: gym.spaces.dict.Dict, curr_obs: gym.spaces.dict.Dict, action
    ):
        raise NotImplementedError()

    def on_task_change(self):
        pass

    def on_new_episode(self):
        pass

    def on_scene_change(self):
        pass

    def action_space_mapping(self, action: int) -> str:
        # map task's actions to `habitat_sim` actions_key, implementing A(s) using habitat_sim registry
        # override to assign custom meaning to your action space or simply to change order of actions
        if action not in self.action_space:
            raise Exception(
                f"action {action} is not a valid action! (Action space: {self.action_space})."
            )
        return self._action_space_map[str(action)]


from avalanche_lab.registry import AvalancheRegistry
import habitat_sim
registry = AvalancheRegistry()

@habitat_sim.registry.register_move_fn(body_action=False)
class NoOp(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec):
        pass


@registry.register_action_params
class NoOpSpec:
    action_key: str = "no_op"

# Signatures are important: use max_steps: np.float=np.float('inf') as :int=np.float('inf') 
# will eval to None and that doesnt go well with config system
@registry.register_task
class VoidTask(Task):
    reward_range = (0.0, 0.0)
    action_space = gym.spaces.discrete.Discrete(4)
    steps: int = 0
    # add args kwargs to avoid rasing errors when random keywords are passed from config
    def __init__(self, sim: Simulator, max_steps: int=100, *args, **kwargs) -> None:
        super().__init__(sim, *args, **kwargs)
        self.max_steps = max_steps
        actions_key = ["no_op", "turn_right", "turn_left", "move_forward"]
        self._action_space_map = {str(i): actions_key[i] for i in range(4)}

    def on_new_episode(self):
        self.steps = 0
        
    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        self.steps += 1
        return False if self.steps < self.max_steps else True

    def reward_function(
        self, prev_obs: gym.spaces.dict.Dict, curr_obs: gym.spaces.dict.Dict, action
    ):
        return 0.0

    def action_space_mapping(self, action: int) -> str:
        # you can access available action using the registry `habitat_sim.registry._mapping['move_fn'].keys()`
        if action not in self.action_space:
            logging.warning(
                f"action {action} is not a valid action! (Action space: {self.action_space}). Default to `no_op` action."
            )
            # default to no_op action
            action = 0
        return self._action_space_map[str(action)]


class Difficulty(enum.IntEnum):
    TRIVIAL = 1.0
    EASY = 1.5
    NORMAL = 2.
    HARD = 3.0


from avalanche_lab.tasks.navigation import *
# all tasks arguments MUST have defaults in order to be passed 
# on to the configuration system. Also, make sure to register task before 
# creating a config
@registry.register_task
class ObjectNav(Task):
    difficulty: Difficulty
    goal: NavigationGoal

    def __init__(
        self,
        sim: Simulator,
        object_asset: str = '',
        difficulty: Difficulty = Difficulty.NORMAL,
        pre_compute_episodes: int = 1,
        keep_goal_fixed: bool = False,
        goal_tollerance: float = .5,
        *args, **kwargs
    ) -> None:
        super().__init__(sim, *args, **kwargs)
        self.sim = sim
        self.difficulty = difficulty
        self.keep_goal_fixed = keep_goal_fixed
        self.n_episodes = pre_compute_episodes
        self.goals = []
        self.tollerance = goal_tollerance

    def on_new_episode(self):
        if not self.keep_goal_fixed:
            return self._generate_goal()

    def on_task_change(self):
        # return self.on_new_episode()
        pass

    def on_scene_change(self):
        # new scene, invalidate generated goals
        self.goals = []

    def _generate_goal(self):
        if not len(self.goals):
            agent = self.sim.get_agent(0)
            # TODO: generate from current position?
            agent_state = agent.get_state()
            self.goals = generate_pointnav_episode(
                self.sim,
                agent_state.position,
                number_of_episodes=self.n_episodes,
                geodesic_to_euclid_starting_ratio=self.difficulty.value,
            )
            if self.goals is None:
                # TODO: Fallback to random point?
                raise Exception("Can't generate new goal")
            
        self.goal = self.goals.pop()

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        # TODO: check if observation space contains agent's position
        # otherwise simply get it from simulator (abs position is unknown to agent)
        agent_pos = self.sim.get_agent(0).get_state().position
        return np.linalg.norm(self.goal.goal_position - agent_pos) < self.tollerance

    def reward_function(self, prev_obs: gym.spaces.dict.Dict, curr_obs: gym.spaces.dict.Dict, action):
        # TODO: not efficient
        return 0. if self.goal_test(None) else -1. 