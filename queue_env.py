import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers.time_limit import TimeLimit

class QueueEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, arrival_rates=[0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.1375, 0.1625, 0.1875], max_queue_len=500):
        self.M = len(arrival_rates)
        self.arrival_rates = arrival_rates
        self.max_queue_len = max_queue_len

        self.observation_space = spaces.MultiDiscrete([self.max_queue_len + 1] * self.M)

        # We have M actions, corresponding to serving each customer
        self.action_space = spaces.Discrete(self.M)

    def _get_obs(self): 
        return np.copy(self._queues)
    
    def _arrive(self): 
        self._queues += self._samples[self.curr, :]
        self.curr += 1
        self._queues[self._queues > self.max_queue_len] = self.max_queue_len
    
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)

        self._queues = np.zeros(self.M, dtype=np.int32)

        observation = self._get_obs()

        self._samples = np.random.poisson(lam=np.tile(self.arrival_rates, (self.max_queue_len, 1)))
        self.curr = 0

        return observation, None
    
    def step(self, action): 
        self._arrive()
        reward = np.zeros(self.M, dtype=np.int32)
        if self._queues[action] > 0:
            self._queues[action] -= 1
            reward[action] = 1
        
        new_observation = self._get_obs()
        return new_observation, reward, False, False, None

def get_env(max_queue_len=500, arrival_rates=[0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.1375, 0.1625, 0.1875], H=500):
    return TimeLimit(QueueEnv(arrival_rates=arrival_rates, max_queue_len=max_queue_len), max_episode_steps=H)