import magnum as mn
import numpy as np
import quaternion  # noqa: F401
from dataclasses import dataclass

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

# more at https://github.com/facebookresearch/habitat-sim/blob/master/examples/tutorials/new_actions.py
# FIXME: this only works programatically for now
@dataclass
class ActionParameters:
    def __post_init__(self):
        # TODO: add current action parameters to configuration
        pass

def add_action(
        agent_cfg: habitat_sim.AgentConfiguration,
        action_key: str,
        action_params,
        action_impl_class: habitat_sim.SceneNodeControl,
        action_name: str,
        body_action: bool = True,
    ):
        """[summary]

        Args:
            agent_cfg (habitat_sim.AgentConfiguration): [description]
            action_key (str): [description]
            action_params ([type]): [description]
            action_impl_class (habitat_sim.SceneNodeControl): [description]
            action_name (str): [description]
            body_action (bool, optional): Whether it affects body of the agent, if False affects only sensors. Defaults to True.
        """
        # register action implementation to simulator registry so that it knows how to act in the sim
        habitat_sim.registry.register_move_fn(
            action_impl_class, name=action_name, body_action=body_action
        )
        # add new action to agent's action space
        agent_cfg.action_space[action_key] = habitat_sim.ActionSpec(
            action_name, action_params
        )
        return agent_cfg

# We will define an action that moves the agent and turns it by some amount
# First, define a class to keep the parameters of the control
@dataclass
class MoveAndSpinSpec:
    forward_amount: float
    spin_amount: float

# Register the control functor
# This action will be an action that effects the body, so body_action=True
@habitat_sim.registry.register_move_fn(body_action=True)
class MoveForwardAndSpin(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: MoveAndSpinSpec
    ):
        forward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.FRONT
        )
        scene_node.translate_local(forward_ax * actuation_spec.forward_amount)

        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.UP
        scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

# Let's define a strafe action!
@dataclass
class StrafeActuationSpec:
    forward_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    strafe_angle: float = 90.0

def _strafe_impl(
    scene_node: habitat_sim.SceneNode, forward_amount: float, strafe_angle: float
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    rotation = quat_from_angle_axis(np.deg2rad(strafe_angle), habitat_sim.geo.UP)
    move_ax = quat_rotate_vector(rotation, forward_ax)

    scene_node.translate_local(move_ax * forward_amount)

@habitat_sim.registry.register_move_fn(body_action=True)
class StrafeLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: StrafeActuationSpec
    ):
        _strafe_impl(
            scene_node, actuation_spec.forward_amount, actuation_spec.strafe_angle
        )

@habitat_sim.registry.register_move_fn(body_action=True)
class StrafeRight(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: StrafeActuationSpec
    ):
        _strafe_impl(
            scene_node, actuation_spec.forward_amount, -actuation_spec.strafe_angle
        )

# This is wrapped in a such that it can be added to a unit test
def main():

    # The habitat_sim.ActionSpec defines an action.  The first arguement is the regsitered name
    # of the control spec, the second is the parameter spec
    # agent_config.action_space["fwd_and_spin"] = habitat_sim.ActionSpec(
        # "move_forward_and_spin", MoveAndSpinSpec(1.0, 45.0)
    # )

    

    # Take the new actions using `action key`
    # sim.step("fwd_and_spin")
    # print(sim.get_agent(0).state)

    

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.action_space["strafe_left"] = habitat_sim.ActionSpec(
        "strafe_left", StrafeActuationSpec(0.25)
    )
    agent_config.action_space["strafe_right"] = habitat_sim.ActionSpec(
        "strafe_right", StrafeActuationSpec(0.25)
    )    
