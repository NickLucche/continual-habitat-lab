import habitat_sim
# from https://aihabitat.org/docs/habitat-sim/rigid-object-tutorial.html#kinematic-object-placement

# remove objects you added
def remove_all_objects(sim):
    for id_ in sim.get_existing_object_ids():
        sim.remove_object(id_)

agent_transform = place_agent(sim)

# get the primitive assets attributes manager
prim_templates_mgr = sim.get_asset_template_manager()

# get the physics object attributes manager
obj_templates_mgr = sim.get_object_template_manager()