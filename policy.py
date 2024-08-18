import torch
import torch.nn as nn
from cfg import Config

class Policy(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Policy, self).__init__()
        n_hidden = Config.Policy.hidden_size
        self.affine1 = nn.Linear(n_states, n_hidden)
        # self.dropout1 = nn.Dropout(p=0.6)
        # self.hidden = nn.Linear(128, 128)
        # self.dropout2 = nn.Dropout(p=0.4)
        self.hidden = nn.Linear(128, 128)
        
        self.affine2 = nn.Linear(n_hidden, n_actions)

        self.optimizer = Config.Policy.optimizer(self.parameters(), lr=Config.Policy.learning_rate)
        
    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout1(x)
        # x = Config.Policy.activation(x)
        # x = self.hidden(x)
        # x = self.dropout2(x)
        x = Config.Policy.activation(x)
        action_scores = self.affine2(x)
        return Config.Policy.parameterization(action_scores, dim=-1)
    
    def sample_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)
        
    def get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data.clone()
        return params

    def set_params(self, params):
        for name, param in self.named_parameters():
            if name in params:
                param.data.copy_(params[name])
            else:
                raise ValueError(f"Parameter {name} not found in provided parameters")
