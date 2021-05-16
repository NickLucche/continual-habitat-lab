# easily set up tasks in any scene without so much as a sweat
from typing import Dict
from habitat_sim import Simulator
import gym.spaces.dict
import gym.spaces.discrete
from gym.spaces.space import Space
from habitat_sim.registry import registry
import logging


class Task:
    sim: Simulator
    reward_range = (-float("inf"), float("inf"))
    # action and obs space as defined by OpenAI Gym
    action_space: Space
    # observation_space: Space TODO: ignore for now

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        raise NotImplementedError()

    def reward_function(
        self, prev_obs: gym.spaces.dict.Dict, curr_obs: gym.spaces.dict.Dict, action
    ):
        raise NotImplementedError()

    def action_space_mapping(self, action: int) -> str:
        # map task's actions to `habitat_sim` actions_key, implementing A(s) using habitat_sim registry
        # override to assign custom meaning to your action space or simply to change order of actions
        raise NotImplementedError()


from avalanche_lab.registry import registry
import habitat_sim
@habitat_sim.registry.register_move_fn(body_action=False)
class NoOp(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec
    ):
        pass
@registry.register_action_params
class NoOpSpec:
    action_key: str = 'no_op'

@registry.register_task
class VoidTask(Task):
    reward_range = (0.0, 0.0)
    action_space = gym.spaces.discrete.Discrete(4)

    def __init__(self, sim: Simulator) -> None:
        super().__init__(sim)
        actions_key = ["no_op", "turn_right", "turn_left", "move_forward"]
        self._action_space_map = {str(i): actions_key[i] for i in range(4)}

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        return False

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


# TODO: PointNav