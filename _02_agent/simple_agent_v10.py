import torch.nn as nn
from gym import Env
from _02_agent.base_agent import AgentBase

class SimpleNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, hidden_layers:int = 1):
        super(SimpleNet, self).__init__()

        modules = []
        modules.append(nn.Linear(obs_size, hidden_size))
        modules.append(nn.ReLU())

        for i in range(0, hidden_layers-1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(hidden_size, n_actions))

        self.net = nn.Sequential(*modules)


    def forward(self, x):
        return self.net(x.float())

class DuelingNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, hidden_layers:int = 1):
        super(DuelingNet, self).__init__()

        self.feauture_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )


    def forward(self, x):
        features = self.feauture_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

import gym
import ptan
import torch
from torch import device

class SimpleAgentV10(AgentBase):

    def __init__(self, env: Env,
                 devicestr:str,
                 gamma:float,
                 buffer_size:int,
                 target_net_sync:int = 1000,
                 eps_start:float = 1.0,
                 eps_final:float = 0.02,
                 eps_frames:int = 10**5,
                 hidden_size:int = 128,
                 hidden_layers:int = 1):

        super(SimpleAgentV10, self).__init__(env, devicestr)

        self.target_net_sync = target_net_sync

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.net = self._config_net()
        print(self.net)

        self.tgt_net = ptan.agent.TargetNet(self.net)

        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            epsilon=1,
            selector=ptan.actions.ArgmaxActionSelector())

        self.epsilon_tracker = ptan.actions.EpsilonTracker(selector=self.selector, eps_start=eps_start, eps_final=eps_final, eps_frames=eps_frames)

        self.agent = agent = ptan.agent.DQNAgent(self.net, self.selector, device = self.device)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.agent, gamma=gamma)
        self.buffer = ptan.experience.ExperienceReplayBuffer(self.exp_source, buffer_size=buffer_size)


    def _config_net(self)-> nn.Module:
        return SimpleNet(self.env.observation_space.shape[0], self.hidden_size, self.env.action_space.n,
                         hidden_layers=self.hidden_layers).to(self.device)


    def iteration_completed(self, iteration: int):

        self.epsilon_tracker.frame(iteration)

        if iteration % self.target_net_sync == 0:
            self.tgt_net.sync()

    def get_net(self):
        return self.net

    def get_tgtnet(self):
        return self.tgt_net

    def get_buffer(self):
        return self.buffer
