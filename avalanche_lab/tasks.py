# easily set up tasks in any scene without so much as a sweat
from habitat_sim import Simulator
import gym.spaces.dict

# import attr

# @attr.s
class Task:
    sim: Simulator
    reward_range = (-float("inf"), float("inf"))
    # TODO: action and obs space

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim

    def initialize_task(self):
        pass

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        raise NotImplementedError()

    def reward_function(self, obs: gym.spaces.dict.Dict, action):
        raise NotImplementedError()

from avalanche_lab.registry import registry
@registry.register_task
class VoidTask(Task):
    reward_range = (0.0, 0.0)

    def __init__(self, sim: Simulator) -> None:
        super().__init__(sim)

    def goal_test(self, obs: gym.spaces.dict.Dict) -> bool:
        return False

