import torch
import numpy as np
from cfg import Config
from policy import Policy
from utils import discounted_sum

class PGMethods:
    def __init__(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        class_name = self.__class__.__name__
        nested = getattr(Config, class_name)

        env_config = Config.env_config
        self.env = Config.env

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.policy = Policy(n_states, n_actions)
        self.gamma = env_config.gamma
        self.single_objective = env_config.single_objective

        self.num_episodes_per_train_step = nested.num_episodes_per_train_step
        self.num_episodes_eval = nested.num_episodes_eval
        
        self.derivative_scalarization = Config.env_config.derivative_scalarization
        self.scalarization = Config.env_config.scalarization

    def collect_full_episode_data(self):
        states = []
        actions = []
        rewards = []
        log_probs = []

        end_episode = False
        state, _ = self.env.reset(seed=self.seed)
        self.seed += 1

        while not end_episode:
            action, log_prob = self.policy.sample_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            end_episode = terminated or truncated

        return states, actions, rewards, log_probs
    
    def collect_single_episode_data(self):
        _, _, rewards, log_probs = self.collect_full_episode_data()
        return rewards, log_probs
    
    def evaluate(self): 
        with torch.no_grad():
            Js = []
            for _ in range(self.num_episodes_eval): 
                rewards, _ = self.collect_single_episode_data()
                Js.append(discounted_sum(rewards, self.gamma))
            return self.scalarization(np.mean(Js, axis=0))
    