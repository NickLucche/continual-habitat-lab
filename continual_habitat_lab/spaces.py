# define obs spaces from gym
class EmptySpace(Space):
    """
    A ``gym.Space`` that reflects arguments space for action that doesn't have
    arguments. Needed for consistency ang always samples `None` value.
    """

    def sample(self):
        return None

    def contains(self, x):
        if x is None:
            return True
        return False

    def __repr__(self):
        return "EmptySpace()"


class ActionSpace(gym.spaces.Dict):
    """
    A dictionary of ``EmbodiedTask`` actions and their argument spaces.

    .. code:: py

        self.observation_space = spaces.ActionSpace({
            "move": spaces.Dict({
                "position": spaces.Discrete(2),
                "velocity": spaces.Discrete(3)
            }),
            "move_forward": EmptySpace(),
        })
    """

    def __init__(self, spaces: Union[List, Dict]):
        if isinstance(spaces, dict):
            self.spaces = OrderedDict(sorted(spaces.items()))
        if isinstance(spaces, list):
            self.spaces = OrderedDict(spaces)
        self.actions_select = gym.spaces.Discrete(len(self.spaces))

    @property
    def n(self) -> int:
        return len(self.spaces)

    def sample(self):
        action_index = self.actions_select.sample()
        return {
            "action": list(self.spaces.keys())[action_index],
            "action_args": list(self.spaces.values())[action_index].sample(),
        }

    def contains(self, x):
        if not isinstance(x, dict) or "action" not in x:
            return False
        if x["action"] not in self.spaces:
            return False
        if not self.spaces[x["action"]].contains(x.get("action_args", None)):
            return False
        return True

    def __repr__(self):
        return (
            "ActionSpace("
            + ", ".join([k + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )