import gymnasium as gym 
import numpy as np 
import gymnasium_robotics
import torch 
import torch.nn as nn 
import torchvision.transforms as T
import torch.hub

class ShadowHandCubeRotation_ImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_joint, n_tactile, device='cuda'):
        super().__init__(env)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.resnet = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.resnet.fc = nn.Identity()
        self.resnet.eval().to(self.device)

        for param in self.resnet.parameters():
            param.requires_grad = False

        #Preprocessing but not resizing to lower resolution to maintain finer detail, only tensor conversion and normalization
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
        ])

        self.n_joint = n_joint
        self.n_resnet_features = 2048
        self.n_tactile = n_tactile

        low = np.full(self.n_joint+self.n_resnet_features+self.n_tactile,-np.inf, dtype=np.float32)
        high = np.full(self.n_joint+self.n_resnet_features+self.n_tactile,np.inf, dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": self.env.observation_space['achieved_goal'],
            "desired_goal": self.env.observation_space['desired_goal'],
        })

    def observation(self, obs):
        joint_states = obs['observation'][:self.n_joint]
        tactile_readings = obs['observation'][-self.n_tactile:]

        img = self.render()
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet(img_tensor).flatten().cpu().numpy()

        new_obs = np.concatenate([joint_states, features, tactile_readings], axis=0)

        return {
            "observation": new_obs,
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"],
        }
