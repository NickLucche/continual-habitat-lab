import habitat_sim
from typing import List, Tuple, Dict, Sequence, Union, Generator
import numpy as np
import logging
from habitat_sim.nav import GreedyGeodesicFollower
from habitat_sim.errors import GreedyFollowerError
from dataclasses import dataclass

# adapted from https://github.com/facebookresearch/habitat-lab/blob/master/habitat/datasets/pointnav/pointnav_generator.py
ISLAND_RADIUS_LIMIT = 1.5


@dataclass
class NavigationGoal:
    source_position: np.ndarray
    source_rotation: np.ndarray
    goal_position: np.ndarray
    shortest_path: List[str]
    geodesic_distance: float
    _num_iterations_to_find: int = None


def get_shortest_path_actions(
    sim: habitat_sim.Simulator,
    source_position: List[float],
    source_rotation: List[Union[int, np.float64]],
    goal_position: List[float],
    success_distance: float = 0.05,
    reset_agent=True,
) -> List[str]:
    """ Find the shortest path from source to target position assuming to have an
    agent with basic default actions available in its actions space 
    (`move_forward`, `turn_left`, turn_right). If not so, this will error.
    For the purpose of this utility function, we're also assuming no actuation noise
    so the returned path will be the one under 'optimal' conditions.

    Args:
        sim (habitat_sim.Simulator): [description]
        source_position (List[float]): [description]
        source_rotation (List[Union[int, float64]]): [description]
        goal_position (List[float]): [description]
        success_distance (float, optional): [description]. Defaults to 0.05.
        max_episode_steps (int, optional): [description]. Defaults to 500.

    Returns:
        List[str]: List of actions' keys taken during the shortest path.
    """

    if reset_agent:
        agent = sim.get_agent(0)
        # state = agent.get_state()
        # reassignment doesnt work here we must re-create agent state for habitat sim
        agent_state = habitat_sim.AgentState()
        agent_state.position = source_position
        agent_state.rotation = source_rotation
        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        # state.sensor_states = {}
        # agent.set_state(state, reset_sensors=True)
        agent.set_state(agent_state)
        # sim.reset()
        
    follower = GreedyGeodesicFollower(
        sim.pathfinder,
        sim.get_agent(0),  # single agent setting
        success_distance,
        fix_thrashing=True,
        thrashing_threshold=16,  # number of actions to stop being a rumba in a corner
    )
    shortest_path = None
    try:
        shortest_path = follower.find_path(goal_position)
    except GreedyFollowerError:
        logging.error(
            f"Unable to find shortest path to target position {goal_position} from {source_position}"
        )

    return shortest_path


def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: habitat_sim.Simulator,
    further_than: float,
    closer_than: float,
    geodesic_to_euclid_ratio: float,
) -> Tuple[bool, float]:
    euclid_dist = np.linalg.norm(np.array(s) - np.array(t))
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0.0
    # compute geosedic distance between start and end point
    path = habitat_sim.ShortestPath()
    path.requested_start = s
    path.requested_end = t
    sim.pathfinder.find_path(path)
    dist = path.geodesic_distance
    if dist == np.inf:
        return False, dist
    if not further_than <= dist <= closer_than:
        return False, dist
    distances_ratio = dist / euclid_dist
    # print("ratio", distances_ratio)
    if distances_ratio < geodesic_to_euclid_ratio:
        return False, dist
    return True, dist


def generate_pointnav_episode(
    sim: habitat_sim.Simulator,
    agent_position: np.ndarray = None,
    generate_shortest_path: bool = True,
    number_of_episodes: int = 1,
    shortest_path_success_distance: float = 0.2,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 50,
    geodesic_to_euclid_starting_ratio: float = 2.0,
    geodesic_to_euclid_min_ratio: float = 1.0,
    number_retries_per_target: int = 100,
    number_retries_per_source: int = 10,
) -> Generator[NavigationGoal, None, None]:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    MAX_SOURCE_SAMPLING = number_retries_per_source

    gte_ratios = np.linspace(
        geodesic_to_euclid_starting_ratio,
        geodesic_to_euclid_min_ratio,
        number_retries_per_target,
    )
    
    # gte_ratios[:number_retries_per_target//5] = gte_ratios[0]
    # gte_ratios[-number_retries_per_target//5:] = gte_ratios[-1]
    goals = []
    for _ in range(number_of_episodes):
        # query NavMesh navigable area using PathFinder API
        if not sim.pathfinder.is_loaded:
            raise Exception(
                "Pathfinder not initialized, unable to sample navigable points."
            )
        pathfinder = sim.pathfinder
        found = False
        if agent_position is None:
            for _ in range(MAX_SOURCE_SAMPLING):
                # first sample source position if not given
                source_position = pathfinder.get_random_navigable_point()
                # print("source", source_position)

                # make sure sampled point is in some 'interesting area' greater than some threshold
                # e.g. avoid sampling a point on top of some other scene object
                if (
                    agent_position is None
                    and pathfinder.island_radius(source_position) < ISLAND_RADIUS_LIMIT
                ):
                    continue
                else:
                    found = True
                    break
            if not found:
                logging.error("Unable to sample valid starting position")
                return None
        else:
            source_position = agent_position
            
        # then sample object goal position making sure "it's not too easy" or undoable
        for _retry in range(number_retries_per_target):
            # try multiple times, starting with a high `geodesic_to_euclid_ratio` and
            # ~linearly decreasing it up to `geodesic_to_euclid_ratio_min` (~min episode difficulty)
            target_position = pathfinder.get_random_navigable_point()
            if sim.pathfinder.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
                continue
            # print("target pos", target_position)

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                further_than=closest_dist_limit,
                closer_than=furthest_dist_limit,
                geodesic_to_euclid_ratio=gte_ratios[_retry],
            )
            # print("distance returned", dist, "current gte", gte_ratios[_retry])
            if is_compatible:
                break
        if is_compatible:
            # sample random orientation
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            # have a shortest path agent try and traverse the path, useful to compute
            # reward based on best trajectory
            shortest_paths = None
            if generate_shortest_path:
                shortest_paths = get_shortest_path_actions(
                    sim,
                    source_position=source_position,
                    source_rotation=source_rotation,
                    goal_position=target_position,
                    success_distance=shortest_path_success_distance,
                )

            goals.append(
                NavigationGoal(
                    source_position,
                    source_rotation,
                    target_position,
                    shortest_paths,
                    dist,
                    _num_iterations_to_find=_retry,
                )
            )
        else:
            logging.error(
                f"Unable to generate a path for current scene with provided configuration (min dist: {closest_dist_limit}, max dist: {furthest_dist_limit}, min gte ratio: {geodesic_to_euclid_min_ratio})"
            )
    return goals

