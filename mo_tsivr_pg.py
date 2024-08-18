from mo_pg import MOPG
from utils import discount_rewards, discounted_sum, get_grad, set_grad, grads_to_1d_tensor
import torch
from cfg import Config
import numpy as np
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_


class MOTSIVRPG(MOPG):
    def __init__(self, seed=42):
        super().__init__(seed)
        self.N = Config.MOTSIVRPG.N
        self.B = Config.MOTSIVRPG.B
        self.n_iterations = Config.MOTSIVRPG.n_iterations
        self.proj = lambda x: x
        self.update_ref_policy()
        
    def apply_loss_ref(self, rewards, log_probs):
        return self.apply_loss_to(rewards, log_probs, self.ref_policy)
        
    def update_ref_policy(self):
        self.ref_policy = deepcopy(self.policy)
    
    def big_iteration(self):
        n = self.N
        #compute derivative of scalarization
        with torch.no_grad():
            trajectory = [self.collect_full_episode_data() for _ in range(n)]
            J = self.compute_scalarization(trajectory)
            self.past_J = J
            derivative_scalarization = np.array(self.derivative_scalarization(J))

        #compute discounted rewards 
        trajectory = [self.collect_full_episode_data() for _ in range(n)]
        self.past_g = self.compute_grad(trajectory, derivative_scalarization)
            
        clip_grad_norm_(self.policy.parameters(), Config.MOTSIVRPG.delta/Config.Policy.learning_rate)
        self.update_ref_policy()

        self.policy.optimizer.step()
        
    def compute_isw(self, trajectory):
        with torch.no_grad():
            isws = []
            
            for states, actions, _, log_probs_policy in trajectory:
                m1 = log_probs_policy
                
                m2 = [torch.distributions.Categorical(self.ref_policy.forward(torch.from_numpy(state).float().unsqueeze(0))).log_prob(action) for state, action in zip(states, actions)]
                
                last = torch.cumsum(torch.Tensor(m1) - torch.Tensor(m2), dim=0)
                isws.append(torch.exp(last))
            
            return isws
     
    def compute_scalarization_is(self, trajectory):
        isw = self.compute_isw(trajectory)
        Js = []

        for (_, _, rewards_single_episode, _), w in zip(trajectory, isw):
            Js.append(self.proj(discounted_sum([r * wt.item() for r, wt in zip(rewards_single_episode, w)], self.gamma)))
        
        J = np.mean(Js, axis=0)
        return J
    
    def compute_grad_is(self, trajectory, derivative_scalarization):
        discounted_rewards_all_episodes = []
        log_probs_all_episodes = []
        
        isw = self.compute_isw(trajectory)

        for (states, actions, rewards_single_episode, _), w in zip(trajectory, isw):
            log_probs_single_episode = [torch.distributions.Categorical(self.policy.forward(torch.from_numpy(state).float().unsqueeze(0))).log_prob(action) for state, action in zip(states, actions)]
            rewards_single_episode = np.sum(rewards_single_episode * derivative_scalarization, axis=1)

            discounted_rewards = discount_rewards(torch.tensor(rewards_single_episode), self.gamma)
            
            discounted_rewards_all_episodes.append(discounted_rewards)
            log_probs_all_episodes.append(torch.cat(log_probs_single_episode) * w)
            
        self.apply_loss_ref(discounted_rewards_all_episodes, log_probs_all_episodes)

        return get_grad(self.ref_policy)
           
    def small_iteration(self):
        n = self.B
        
        #compute derivative of scalarization
        with torch.no_grad():
            trajectory = [self.collect_full_episode_data() for _ in range(n)]
            J_new = self.compute_scalarization(trajectory)
            J_is = self.compute_scalarization_is(trajectory)
            J = self.proj(J_new - J_is + self.past_J)
            derivative_scalarization = np.array(self.derivative_scalarization(J))
            
        trajectory = [self.collect_full_episode_data() for _ in range(n)]
        g_new = self.compute_grad(trajectory, derivative_scalarization)
        g_is = self.compute_grad_is(trajectory, np.array(self.derivative_scalarization(self.past_J)))
        g = {name: g_new[name] - g_is[name] + self.past_g[name] for name in g_new.keys()}
        self.past_J = J
        self.past_g = g
        
        set_grad(self.policy, grad_dict=g)
            
        clip_grad_norm_(self.policy.parameters(), Config.MOTSIVRPG.delta/Config.Policy.learning_rate)
        self.update_ref_policy()
        self.policy.optimizer.step()

    def train_step(self):
        self.big_iteration()
        
        for _ in range(self.n_iterations):
            self.small_iteration()

        return self.policy
