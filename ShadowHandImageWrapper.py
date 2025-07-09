import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torchvision.transforms as T

class ShadowHandImageWrapper_VisualGoal(gym.ObservationWrapper):
    def __init__(self, env, use_tactile=True, device='cuda'):
        super().__init__(env)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load frozen ResNet-50
        self.resnet = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.resnet.fc = nn.Identity()
        self.resnet.eval().to(self.device)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Normalize and convert to tensor
        self.preproc = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        self.use_tactile = use_tactile
        self.n_tactile = 92 if use_tactile else 0
        self.n_resnet_feat = 2048
        self.kinematic_indices = np.arange(48)

        total_dim = len(self.kinematic_indices) + self.n_resnet_feat + self.n_resnet_feat + self.n_tactile

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32),
            "achieved_goal": self.env.observation_space["achieved_goal"],
            "desired_goal": self.env.observation_space["desired_goal"],
        })

    def get_resnet_features(self, img):
        img_tensor = self.preproc(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.resnet(img_tensor).flatten().cpu().numpy()
        return feat

    def render_goal_image(self, goal_pose):
        # Save current object pose
        original_pose = self.env.sim.data.get_joint_qpos("object:joint").copy()

        # Set cube to goal pose
        self.env.sim.data.set_joint_qpos("object:joint", goal_pose)
        self.env.sim.forward()

        img = self.env.render()

        # Restore original object pose
        self.env.sim.data.set_joint_qpos("object:joint", original_pose)
        self.env.sim.forward()

        return img

    def observation(self, obs):
        state = obs["observation"]
        kinematic = state[self.kinematic_indices]
        tactile = state[-self.n_tactile:] if self.use_tactile else np.array([])

        # Current image
        current_img = self.render()
        current_feat = self.get_resnet_features(current_img)

        # Goal image
        goal_pose = obs["desired_goal"]
        goal_img = self.render_goal_image(goal_pose)
        goal_feat = self.get_resnet_features(goal_img)

        combined_obs = np.concatenate([kinematic, current_feat, goal_feat, tactile], axis=0)

        return {
            "observation": combined_obs,
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"]
        }
