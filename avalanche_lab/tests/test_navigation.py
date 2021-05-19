import os, logging

# remove info logging
os.environ["GLOG_minloglevel"] = "3"
import habitat_sim
from habitat_sim.logging import logger

# logger.disabled = True
# logger.propagate = False
# logger = logging.getLogger(habitat_sim.logging.__file__)
# print('filename', habitat_sim.logging.__file__)
from avalanche_lab.tasks.navigation import generate_pointnav_episode
from avalanche_lab.config import AvalancheConfig
import pytest
import numpy as np


@pytest.mark.parametrize(
    "gte_ratio, min_gte_ratio, diff",
    [(1.0, 1.0, 0), (1.1, 1.0, 1), (1.4, 1.0, 2), (1.8, 1.2, 3)],
)
def test_difficulties_episode_gen(gte_ratio, min_gte_ratio, diff: int):
    # test scenes are very easy and have almost no obstacles in them
    sceneids = [
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        # we'll need a more complex scene to have a higher chance of getting harder episode
        "/home/nick/datasets/habitat/gibson/gibson/Cokeville.glb",
        "/home/nick/datasets/habitat/gibson/gibson/Cokeville.glb",
        "/home/nick/datasets/habitat/gibson/gibson/Cokeville.glb",
    ]
    scene = sceneids[diff]
    if not os.path.exists(scene):
        pytest.skip(f"Must download scene {scene} to run this test!")
    cfg = AvalancheConfig(from_cli=False)
    with habitat_sim.Simulator(cfg.make_habitat_sim_config(scene)) as sim:
        sim.initialize_agent(0)
        agent_pos = sim.get_agent(0).get_state().position
        obs = sim.get_sensor_observations(0)["rgb"]
        eps = generate_pointnav_episode(
            sim,
            number_retries_per_target=100,
            geodesic_to_euclid_starting_ratio=gte_ratio,
            geodesic_to_euclid_min_ratio=min_gte_ratio,
            number_of_episodes=1,
        )
        # check agent position wasn't modified during search
        assert np.linalg.norm(obs - sim.get_sensor_observations(0)["rgb"]) < 0.01
        assert np.linalg.norm(agent_pos - sim.get_agent(0).get_state().position) < 0.1
        num_iterations = [20, 80, 120, 200]
        for ep in eps:
            print(
                "GTE ratio",
                ep.geodesic_distance
                / np.linalg.norm(ep.goal_position - ep.source_position),
            )
            # assert it wasnt too hard finding it
            assert ep._num_iterations_to_find <= num_iterations[diff]
            # assert path not empty
            assert ep.shortest_path is not None

