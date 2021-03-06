from continual_habitat_lab.tasks.tasks import ObjectNav
from omegaconf.omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from argparse import ArgumentParser
import habitat_sim
import os
from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
from gym.wrappers import RecordVideo

# remove info logging
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

# sys.path.insert(0, "/home/nick/uni/thesis/avalanche-lab/")

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"
RANDOM_TP = "t"
SAVE_FRAME = "p"


# def rgb2bgr(image):
# return image[..., ::-1]


def visualize(flag: bool, obs):
    if flag:
        img = obs["rgba"]#.astype(np.uint8)
        # if "semantic" in obs:
        #     # this may merge some classes
        #     sem = (obs['semantic']/obs['semantic'].max()).astype(np.uint8)
        #     print(sem.max(), sem.min(), sem.dtype)
        #     print(img.max(), img.min(), img.dtype)
        #     # rgb sensor returns 4 channels by default (rgba)
        #     img = np.hstack(
        #         [
        #             img,
        #             np.zeros((img.shape[0], 2, 4)),
        #             obs["semantic"][:, :, np.newaxis].repeat(4, axis=2),
        #         ]
        #     )
        #     print('img shape', img.shape)
        print(img.shape, obs.keys())
        if 'collided' in obs:
            print('collided', obs['collided'])
        print(img[..., 3].max(), img[..., 3].min(), img[..., 3].mean())
        img = img[..., :3].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        cv2.imshow("RGBA", img)
        return img


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "-i", "--interactive", help="Run demo interactively", action="store_true",
    )
    args.add_argument(
        "-n", "--n-episodes", help="Number of episodes to run", type=int, default=3,
    )
    args.add_argument(
        "-r", "--record-video", help="Record episode frames in `video` folder", action="store_true"
    )
    args = args.parse_args()
    n_episodes = args.n_episodes

    # config = ContinualHabitatLabConfig(from_cli=False)
    config = ContinualHabitatLabConfig.from_yaml("example_config.yaml")
    # config.scene.max_scene_repeat_episodes = 1

    print("Simulator configuration:\n", config.to_yaml())
    with ContinualHabitatEnv(config) as env:
        if args.record_video:
            env = RecordVideo(env, 'video', episode_trigger=lambda _: True)
        task_idx = 0
        action_names = list(config.habitat_sim_config.agents[0].action_space.keys())
        print("Available actions", action_names)
        print(
            config.habitat_sim_config.agents[0].action_space,
            type(config.habitat_sim_config.agents[0].action_space),
        )
        print("Current scene:", env.current_scene)
        print("Available Tasks:", env.tasks)
        end = False
        for _ in range(n_episodes):
            obs = env.reset()
            print("Current task:", env.current_task)
            print("Initial position", env.agent_position)

            img = visualize(args.interactive, obs)
            # assert env._curr_task_idx == task_idx, "Task should change at each new episode"
            task_idx = (task_idx + 1) % 2
            step = 0
            while not env.done:
                if isinstance(env.current_task, ObjectNav):
                    print("Goal position", env.current_task.goal.goal_position)

                # for i in range(3):
                if args.interactive:
                    keystroke = cv2.waitKey(0)
                    # ord gets unicode from one-char string
                    if keystroke == ord(FORWARD_KEY):
                        action = "move_forward"
                    elif keystroke == ord(LEFT_KEY):
                        action = "turn_left"
                    elif keystroke == ord(RIGHT_KEY):
                        action = "turn_right"
                    elif keystroke == ord(FINISH):
                        break
                    elif keystroke == ord(SAVE_FRAME):
                        print('saving image', img.shape, img.dtype)
                        cv2.imwrite(f'habitat_{step}.png', img)
                    elif keystroke == ord(RANDOM_TP):
                        agent = env.sim.get_agent(0)
                        agent_state = habitat_sim.AgentState()
                        # state = agent.get_state()
                        pos = env.sim.pathfinder.get_random_navigable_point()
                        agent_state.position = pos
                        # agent_state.sensor_states = {}
                        # env.sim.get_agent(0).set_state(state, reset_sensors=True)
                        agent.set_state(agent_state)  # , reset_sensors=True)
                        # env.sim.reset()
                        # to get new obs
                        action = "no_op"
                    elif keystroke == ord("q") or keystroke == 127:
                        print("Closing..")
                        end = True
                        break
                    else:
                        print("INVALID KEY")
                        continue
                    # TODO:
                    # if action not in env.task.actions:
                    #     print("Invalid action!")
                    #     continue
                else:
                    # follow generated shortest path
                    if isinstance(env.current_task, ObjectNav):
                        action = env.current_task.goal.shortest_path[step]
                    else:
                        # execute random action
                        action = env.action_space.sample()
                    # action = random.choice(action_names)
                # env.step(env.action_space.sample())
                print("action", action)
                obs, reward, done, _ = env.step(action)
                print(f"Reward: {reward}, done: {done}")
                img = visualize(args.interactive, obs)
                print("Current position", env.agent_position)
                step += 1
            if end:
                break
    cv2.destroyAllWindows()
