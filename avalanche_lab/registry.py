#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Registry is central source of truth in Habitat.

Taken from Pythia, it is inspired from Redux's concept of global store.
Registry maintains mappings of various information to unique keys. Special
functions in registry can be used as decorators to register different kind of
classes.

Import the global registry object using

.. code:: py

    from habitat.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a task: ``@registry.register_task``
-   Register a task action: ``@registry.register_task_action``
-   Register a simulator: ``@registry.register_simulator``
-   Register a sensor: ``@registry.register_sensor``
"""

from avalanche_lab.simulator_interactions.building_blocks import ActionParameters
from habitat_sim.agent.agent import _default_action_space
import collections
from typing import Any, Callable, DefaultDict, Optional, Type, Dict, List

from habitat_sim import ActionSpec
from avalanche_lab.tasks.tasks import Task
from habitat_sim.registry import _Registry
import re

def _camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class AvalancheRegistry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    def __init__(self) -> None:
        pass

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_move_fn(
        cls,
        controller: Optional[Type] = None,
        *,
        name: Optional[str] = None,
        body_action: Optional[bool] = None,
    ):
        r"""Registers a new control with Habitat-Sim. Registered controls can
        then be retrieved via `get_move_fn()`

        See `new-actions <new-actions.html>`_ for an example of how to add new actions
        *outside* the core habitat_sim package.

        :param controller: The class of the controller to register. Must inherit from `agent.SceneNodeControl`.
            If :py:`None`, will return a wrapper for use with decorator syntax.
        :param name: The name to register the control with. If :py:`None`, will
            register with the name of the controller converted to snake case,
            i.e. a controller with class name ``MoveForward`` will be registered as
            ``move_forward``.
        :param body_action: Whether or not this action manipulates the agent's body
            (thereby also moving the sensors) or manipulates just the sensors.
            This is a non-optional keyword arguement and must be set (this is done
            for readability).
        """
        assert (
            body_action is not None
        ), "body_action must be explicitly set to True or False"
        from habitat_sim.agent.controls.controls import SceneNodeControl



        name = _camel_to_snake(controller.__name__) if name is None else name
        return cls._register_impl(
            "move_fn", controller(body_action), name, assert_type=SceneNodeControl
        )
        

    @classmethod
    def register_task(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a task to registry with key :p:`name`

        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat.core.registry import registry
            from habitat.core.embodied_task import EmbodiedTask

            @registry.register_task
            class MyTask(EmbodiedTask):
                pass


            # or

            @registry.register_task(name="MyTaskName")
            class MyTask(EmbodiedTask):
                pass

        """

        return cls._register_impl(
            "task", to_register, name, assert_type=Task
        )
    @classmethod
    def register_action_params(cls, to_register=None, *, name: Optional[str] = None):
        r"""

        """
        from avalanche_lab.simulator_interactions.building_blocks import ActionParameters
        # assert issubclass(
        #             to_register, ActionParameters
        #         ), "{} must be a subclass of {}".format(
        #             to_register, ActionParameters
        #         )
        name = to_register.action_key if name is None else name
        return cls._register_impl(
            "action_params", to_register, name
        )



    # @classmethod
    # def register_simulator(
    #     cls, to_register: None = None, *, name: Optional[str] = None
    # ):
    #     r"""Register a simulator to registry with key :p:`name`

    #     :param name: Key with which the simulator will be registered.
    #         If :py:`None` will use the name of the class

    #     .. code:: py

    #         from habitat.core.registry import registry
    #         from habitat.core.simulator import Simulator

    #         @registry.register_simulator
    #         class MySimulator(Simulator):
    #             pass


    #         # or

    #         @registry.register_simulator(name="MySimName")
    #         class MySimulator(Simulator):
    #             pass

    #     """

    #     return cls._register_impl(
    #         "sim", to_register, name, assert_type=Simulator
    #     )

    # @classmethod
    # def register_sensor(cls, to_register=None, *, name: Optional[str] = None):
    #     r"""Register a sensor to registry with key :p:`name`

    #     :param name: Key with which the sensor will be registered.
    #         If :py:`None` will use the name of the class
    #     """

    #     return cls._register_impl(
    #         "sensor", to_register, name, assert_type=Sensor
    #     )

    # @classmethod
    # def register_measure(cls, to_register=None, *, name: Optional[str] = None):
    #     r"""Register a measure to registry with key :p:`name`

    #     :param name: Key with which the measure will be registered.
    #         If :py:`None` will use the name of the class
    #     """

    #     return cls._register_impl(
    #         "measure", to_register, name, assert_type=Measure
    #     )

    # @classmethod
    # def register_task_action(
    #     cls, to_register=None, *, name: Optional[str] = None
    # ):
    #     r"""Add a task action in this registry under key 'name'

    #     :param action_space: An action space that describes parameters to the
    #         task action's method. If :py:`None` then the task action's method
    #         takes no parameters.
    #     :param name: Key with which the task action will be registered. If
    #         :py:`None` will use the name of the task action's method.
    #     """

    #     return cls._register_impl(
    #         "task_action", to_register, name, assert_type=Action
    #     )


    # @classmethod
    # def register_action_space_configuration(
    #     cls, to_register=None, *, name: Optional[str] = None
    # ):
    #     r"""Register a action space configuration to registry with key :p:`name`

    #     :param name: Key with which the action space will be registered.
    #         If :py:`None` will use the name of the class
    #     """

    #     return cls._register_impl(
    #         "action_space_config",
    #         to_register,
    #         name,
    #         assert_type=ActionSpaceConfiguration,
    #     )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_task(cls, name: str) -> Type[Task]:
        return cls._get_impl("task", name)

    @classmethod
    def get_action_params(cls, name: str) -> Type[ActionParameters]:
        return cls._get_impl("action_params", name)

    @classmethod
    def get_all_action_params(cls) -> Dict[str, ActionParameters]:
        return cls.mapping['action_params']

    @classmethod
    def get_move_fn(cls, name: str) -> Type[Task]:
        return cls._get_impl("move_fn", name)

    # @classmethod
    # def get_task_action(cls, name: str) -> Type[Action]:
    #     return cls._get_impl("task_action", name)

    # @classmethod
    # def get_simulator(cls, name: str) -> Type[Simulator]:
    #     return cls._get_impl("sim", name)

    # @classmethod
    # def get_sensor(cls, name: str) -> Type[Sensor]:
    #     return cls._get_impl("sensor", name)

    # @classmethod
    # def get_measure(cls, name: str) -> Type[Measure]:
    #     return cls._get_impl("measure", name)

    # @classmethod
    # def get_dataset(cls, name: str) -> Type[Dataset]:
    #     return cls._get_impl("dataset", name)

    # @classmethod
    # def get_action_space_configuration(
    #     cls, name: str
    # ) -> Type[ActionSpaceConfiguration]:
    #     return cls._get_impl("action_space_config", name)


from habitat_sim.agent.agent import ActuationSpec
registry = AvalancheRegistry()
# initialize registry with default action specs
for k in _default_action_space():
    # spec = v.actuation
    registry.register_action_params(ActuationSpec, name=k)
