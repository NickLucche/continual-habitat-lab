from omegaconf import OmegaConf
import torch
import os

# remove info logging
os.environ["GLOG_minloglevel"] = "3"
import numpy as np
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader
import habitat_sim
from typing import Tuple
from dataclasses import dataclass
from continual_habitat_lab.registry import registry
from continual_habitat_lab.tasks.tasks import ObjectNav, Difficulty, NavigationGoal
from continual_habitat_lab.config import ContinualHabitatLabConfig
from continual_habitat_lab.env import ContinualHabitatEnv
from continual_habitat_lab.tasks.navigation import generate_pointnav_episode
import random
from continual_habitat_lab.logger import chlab_logger
from typing import List
import cv2
from glob import glob
from torchvision.utils import save_image

MAX_RETRIES = 30

# custom task for exploring dataset without rewards/goal_test
@registry.register_task
class SceneExplorer(ObjectNav):
    def __init__(self, sim: habitat_sim.Simulator, *args, **kwargs) -> None:
        super().__init__(sim, *args, **kwargs)

    def goal_test(self, obs) -> bool:
        return False

    def reward_function(self, prev_obs, curr_obs, action):
        return 0.0

    def _generate_random_path(self, n: int):
        goals = []
        # generate random path from sensible starting points
        for i in range(n):
            # if not sim.pathfinder.is_loaded:
            pathf = self.sim.pathfinder
            source = pathf.get_random_navigable_point()
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            n_steps = random.randint(5, 50)
            goals.append(
                NavigationGoal(
                    source,
                    source_rotation,
                    None,
                    np.random.choice(
                        ["move_forward", "turn_right", "turn_left"], n_steps
                    ).tolist()
                    + [None],
                    None,
                    _num_iterations_to_find=None,
                )
            )
        return goals

    def _generate_goal(self):
        if not len(self.goals):
            # self.goals = generate_pointnav_episode(
            #     self.sim,
            #     number_of_episodes=self.n_episodes,
            #     geodesic_to_euclid_starting_ratio=self.difficulty.value,
            #     geodesic_to_euclid_min_ratio=1.0,
            # )
            self.goals = self._generate_random_path(self.n_episodes)
            if self.goals is None:
                raise Exception("Can't generate new goal")

        self.goal = self.goals.pop()
        if self.goal.shortest_path is None:
            raise Exception("")


explorer_config = {
    "tasks": [
        {
            "type": "SceneExplorer",
            "name": "DatasetExplorer",
            "difficulty": Difficulty.TRIVIAL,
        }
    ],
    "agent": {
        "action_space": {
            # you only need to specify the variable to pass to your action parameters
            "move_forward": {"amount": 1.0},
            "turn_right": {"amount": 45.0},
            "turn_left": {"amount": 45.0},
        },
        "sensor_specifications": [
            {"type": "RGB"},
            {"type": "SEMANTIC"},
            {"type": "DEPTH"},
        ],
    },
    "scene": {
        "scene_path": "/home/nick/datasets/habitat/replicav1/room_2/habitat/mesh_semantic.ply"
        # "dataset_paths": ["/home/nick/datasets/habitat/gibson/gibson/"]
        # "dataset_paths": ["/home/nick/datasets/habitat/scene_dataset/mp3d/v1/tasks/mp3d/"]
    },
}

# TODO: gather paths and assign scene id to them without invalidating em by subclassing task
# TODO: CUDA habitat sim to get observations directly on gpu
@dataclass
class VisualExplorationDataset(IterableDataset):
    r"""
    This class abstracts the exploration of a dataset intended as a set of `habitat_sim` 
    navigable scenes as a Pytorch dataset, to be used for learning from images with 
    semantic/depth groundtruth.
    To provide data, we're going to sample different points per scene and have and greedy
    agent traverse the scene gathering observations as it goes.
    """
    config: ContinualHabitatLabConfig = ContinualHabitatLabConfig(
        OmegaConf.create(explorer_config)
    )
    img_resolution: Tuple[int, int] = (128, 128)
    # instance segmentation flag
    semantic: bool = True
    depth: bool = False
    paths_per_scene: int = 10
    to_tensor: bool = True
    instance_segmentation: bool = False
    # manual batching to have better segmentation mapping performance
    batch_size: int = 1
    dataset_path: str = None
    max_batches: int = None

    def __post_init__(self):
        # instatiate an env using ObjectNav task to explore scenes and 'modded' configs
        # iterable dataset workers are replicated on each worker therefore we'll handle multiple envs
        # self.config = OmegaConf.merge(self.config, )
        if self.dataset_path is not None:
            self.config.scene.dataset_paths = [self.dataset_path]
            self.config.scene.scene_path = None

        self.config.scene.max_scene_repeat_episodes = self.paths_per_scene
        self.config.tasks[0].pre_compute_episodes = self.paths_per_scene
        if not self.semantic:
            self.config.agent.sensor_specifications.pop(1)
            if not self.depth:
                self.config.agent.sensor_specifications.pop(1)
        elif not self.depth:
            self.config.agent.sensor_specifications.pop(2)

        for spec in self.config.agent.sensor_specifications:
            spec.resolution = self.img_resolution

        self.step = 0
        self.batches = 0
        print(OmegaConf.to_yaml(self.config._config))
        self.env = ContinualHabitatEnv(self.config)
        self.env.reset()
        # get semantic scene mapping
        scene = self.env.sim.semantic_scene
        # categories list scene.categories[1].index()/name()
        self.category_id_to_name = {
            cat.index(): cat.name() for cat in scene.categories if cat is not None
        }
        self.category_id_to_name[0] = "unknown"
        print(self.category_id_to_name)
        self.object_id_to_category_id = {
            int(obj.id.split("_")[-1]): obj.category.index()
            for obj in scene.objects
            if obj and obj.category is not None
        }
        self.object_id_to_category_id[0] = 0
        self.n_categories = len(self.category_id_to_name)
        # object_id_to_category_name = {int(obj.id.split("_")[-1]): obj.category.name() for obj in scene.objects if obj is not None}
        print(self.object_id_to_category_id)
        # to explore scenes
        # self._pathfinder = self.env.sim.pathfinder

    def _instance_seg_to_category(self, x: np.ndarray):
        for objid in np.unique(x):
            # TODO: investigate
            x[x == objid] = (
                self.object_id_to_category_id[objid]
                if objid in self.object_id_to_category_id
                else 0
            )
        return x

    def __iter__(self):
        # TODO: should implement to work with dataloader
        # worker_info = torch.utils.data.get_worker_info()

        return self

    def __next__(self):
        if self.max_batches is not None and self.batches >= self.max_batches:
            raise StopIteration()
        self.batches += 1
        batched_obs = {"rgb": []}
        if self.semantic:
            batched_obs["semantic"] = []
        if self.depth:
            batched_obs["depth"] = []

        for _ in range(self.batch_size):
            action = self.env.current_task.goal.shortest_path[self.step]
            # print('len of path', len(self.env.current_task.goal.shortest_path))
            self.step += 1
            obs, _, _, _ = self.env.step(action)
            # this is the last step, reset env/get new path
            if self.env.current_task.goal.shortest_path[self.step] is None:
                self.step = 0
                self.env.reset()
            # collision key ignored here
            for k in batched_obs:
                batched_obs[k].append(obs[k])
        # stack obs
        batched_obs = {k: np.stack(v, axis=0) for k, v in batched_obs.items()}
        # ignore alpha channel
        batched_obs["rgb"] = batched_obs["rgb"][..., :3]
        if self.semantic:
            if not self.instance_segmentation:
                batched_obs["semantic"] = self._instance_seg_to_category(
                    batched_obs["semantic"]
                )
            # pytorch does not support uint32
            batched_obs["semantic"] = batched_obs["semantic"].astype(np.int32)

        if self.to_tensor:
            batched_obs["rgb"] = torch.from_numpy(batched_obs["rgb"]).permute(
                0, 3, 1, 2
            )
            if self.semantic:
                batched_obs["semantic"] = torch.from_numpy(batched_obs["semantic"])
            if self.depth:
                batched_obs["depth"] = torch.from_numpy(batched_obs["depth"])
        return batched_obs

    def __exit__(self):
        self.close()

    def close(self):
        self.env.close()


class AsyncVisualExplorationDataset(Dataset):
    """
    Dataset which which first gathers observations from the environment
    and then stores those observations on disk in order to have a classical
    image dataset.
    Train-test split with data from different scenes is supported.
    """

    depth_scaling_factor: int = 50000

    def __init__(
        self,
        root: str,
        size: int = 10000,
        recompute: bool = False,
        subdir_suffix: str = "_train",
        load_semantic: bool = True,
        load_depth: bool = False,
        **explorer_kwargs,
    ) -> None:
        self.dataset_size = size
        self.root = root
        self.dataset_loaded = False
        self.expl_kwargs = explorer_kwargs
        self.subdir_suffix = subdir_suffix
        self.depth = load_depth
        self.semantic = load_semantic
        if not recompute and os.path.exists(
            os.path.join(root, f"avalanche_habitat_dataset{subdir_suffix}")
        ):
            chlab_logger.info(f"Found exisiting dataset folder in {self.root}.")
            self.dataset_loaded = True
            self.dataset_size = len(
                glob(
                    os.path.join(
                        root, f"avalanche_habitat_dataset{subdir_suffix}", "*_rgb.png"
                    )
                )
            )
        else:
            try:
                os.mkdir(
                    os.path.join(root, f"avalanche_habitat_dataset{subdir_suffix}")
                )
                chlab_logger.info(
                    "Pre-exisiting dataset folder not found, creating dataset by exploring environment..."
                )
            except:
                chlab_logger.info("Recomputing dataset..")
            self.create_dataset()
            self.dataset_loaded = True

    def create_dataset(
        self, batch_size: int = 10, num_workers=1, subdir_suffix: str = None
    ):
        subdir_suffix = self.subdir_suffix if subdir_suffix is None else subdir_suffix
        # tensor transform is done by dataloader
        self.expl_kwargs.update({"to_tensor": False, "batch_size": batch_size})
        print("args", self.expl_kwargs)
        # create dataloader to explore env
        dataset = VisualExplorationDataset(**self.expl_kwargs)
        # this will result in multiple ContinualHabitatEnv being created
        # enable manual batching
        # dl = DataLoader(dataset, batch_size=None, num_workers=num_workers) TODO: avoid collate_fn we dont need tensor
        for bidx, obs in enumerate(dataset):
            # print(len(obs), [v.shape for v in obs.values()])
            self._save_tensors_to_images(obs, bidx, batch_size, subdir_suffix)
            if (bidx + 1) * batch_size * len(obs) >= self.dataset_size:
                break
        chlab_logger.info("Done generating samples")

    def _save_tensors_to_images(
        self,
        obs: List[torch.Tensor],
        batch_idx: int,
        batch_size: int,
        subdir_suff: str = "_train",
    ):
        # save images resulting from batched tensors
        for type_, batched in obs.items():
            # scale distances before saving to png
            if type_ == "depth":
                batched = (batched * self.depth_scaling_factor).astype(np.uint16)
            # elif type_ == 'semantic':
            # batched = batched.astype(np.uint16)
            for i, img in enumerate(batched):
                cv2.imwrite(
                    os.path.join(
                        self.root,
                        f"avalanche_habitat_dataset{subdir_suff}/{batch_idx*batch_size+i}_{type_}.png",
                    ),
                    img,
                )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        self.rgb = True
        x = {}
        for type_ in ["rgb", "semantic", "depth"]:
            if getattr(self, type_):
                img_path = os.path.join(
                    self.root,
                    f"avalanche_habitat_dataset{self.subdir_suffix}",
                    f"{index}_{type_}.png",
                )
                x[type_] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if x[type_] is None:
                    raise Exception(
                        f"Unable to load {img_path}, make sure this file exists and that you're loading from the right directory!"
                    )
                if type_ == "rgb":
                    x[type_] = x[type_].transpose(2, 1, 0)
                elif type_ == "depth":
                    x[type_] = (x[type_] / self.depth_scaling_factor).astype(np.float32)

        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # needs to have this file to load semantic annotations
    # Loading Semantic Stage mesh : ../mp3d/v1/tasks/mp3d/ZMojNkEp431/ZMojNkEp431_semantic.ply
    dataset = AsyncVisualExplorationDataset(
        "/home/nick/datasets/",
        recompute=False,
        img_resolution=(512, 512),
        size=100,
        load_depth=True,
        load_semantic=True,
        semantic=True,
        depth=True,
    )
    # for i in range(3):
    #     x = ds[i]
    #     for t in ['rgb', 'semantic', 'depth']:
    #         if t in x:
    #             plt.imshow(x[t])
    #             plt.show()
    # dataset = VisualExplorationDataset(
    #     paths_per_scene=1,
    #     semantic=True,
    #     depth=True,
    #     img_resolution=(512, 512),
    #     batch_size=1,
    # )
    # dl = DataLoader(dataset, batch_size=None)
    dl = DataLoader(dataset, batch_size=5, shuffle=False)
    for i, obs in enumerate(dl):
        for t in ["rgb", "semantic", "depth"]:
            if t in obs:
                print(t, obs[t].shape)
                if t == "rgb":
                    plt.imshow(obs[t][0].squeeze().permute(2, 1, 0).numpy())
                else:
                    plt.imshow(obs[t][0].squeeze().numpy())
                plt.show()
        if i >= 3:
            break
    # id = iter(dataset)
    # for i in range(3):
    #     obs = next(id)
    #     print("current scene", dataset.env.current_scene)
    #     print("num obs", i)
    #     rgb = obs["rgb"]  # .astype(np.uint8)
    #     depth = obs["depth"]  # .astype(np.uint8)
    #     semantic = obs["semantic"]  # .astype(np.uint8)
    #     print("Categories in semantic:", [dataset.category_id_to_name[x] for x in np.unique(semantic)])
    #     print(rgb.shape, rgb.max(), rgb.min(), rgb.dtype)
    #     # dist in meters
    #     print(depth.shape, depth.max(), depth.min(), depth.dtype)
    #     print(semantic.shape, semantic.max(), semantic.min(), semantic.dtype)
    #     # cv2.imshow("rgb", rgb)
    #     plt.imshow(rgb.squeeze().permute(1, 2, 0).numpy())
    #     plt.show()
    #     plt.imshow(depth.squeeze().numpy())
    #     plt.show()
    #     plt.imshow(semantic.squeeze().numpy(), cmap='tab20')
    #     plt.show()
    # key = cv2.waitKey(0)
    # if key == ord("q"):
    # break

    # cv2.destroyAllWindows()

