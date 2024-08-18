from pgmethods import PGMethods
from utils import discount_rewards, discounted_sum, get_grad
import torch
from cfg import Config
import numpy as np


class MOPG(PGMethods):
    def __init__(self, seed=42):
        super().__init__(seed)
        
    def collect_full_episode_data(self):
        states, actions, rs, l = super().collect_full_episode_data()
        if self.single_objective:
            rs = [np.array([r]) for r in rs] 
        return states, actions, rs, l
    
    def apply_loss_to(self, rewards, log_probs, model):
        assert len(rewards) == len(log_probs)

        model.optimizer.zero_grad()
        loss = [torch.sum(-reward * log_prob) for reward, log_prob in zip(rewards, log_probs)]
        loss = torch.stack(loss).mean(axis=0)        
        loss.backward()
        
    def apply_loss(self, rewards, log_probs):
        return self.apply_loss_to(rewards, log_probs, self.policy)

    def compute_grad(self, trajectory, derivative_scalarization):
        discounted_rewards_all_episodes = []
        log_probs_all_episodes = []

        for _, _, rewards_single_episode, log_probs_single_episode in trajectory:
            rewards_single_episode = np.sum(rewards_single_episode * derivative_scalarization, axis=1)
            discounted_rewards = discount_rewards(torch.tensor(rewards_single_episode), self.gamma)
            
            discounted_rewards_all_episodes.append(discounted_rewards)
            log_probs_all_episodes.append(torch.cat(log_probs_single_episode))
            
        self.apply_loss(discounted_rewards_all_episodes, log_probs_all_episodes)

        return get_grad(self.policy)
    
    def compute_scalarization(self, trajectory):
        Js = []

        for _, _, rewards_single_episode, _ in trajectory:
            Js.append(discounted_sum(rewards_single_episode, self.gamma))
        
        J = np.mean(Js, axis=0)
        return J

    def train_step(self):        
        n = self.num_episodes_per_train_step

        #compute derivative of scalarization
        with torch.no_grad():
            trajectory = [self.collect_full_episode_data() for _ in range(n)]
            J = self.compute_scalarization(trajectory)
            derivative_scalarization = np.array(self.derivative_scalarization(J))

        #compute discounted rewards 
        trajectory = [self.collect_full_episode_data() for _ in range(n)]
        self.compute_grad(trajectory, derivative_scalarization)

        self.policy.optimizer.step()
        return self.policy
