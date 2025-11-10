import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import weights_init_


# Gaussian policy
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # state confidence for ILEED
        self.rho_1 = nn.Linear(state_dim, hidden_dim)
        self.rho_2 = nn.Linear(hidden_dim, hidden_dim)
        self.rho_3 = nn.Linear(hidden_dim, 1)

        # helper functions
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

    # state confidence for ILEED
    def rho(self, state):
        x = self.m(self.rho_1(state))
        x = self.m(self.rho_2(x))
        return self.rho_3(x)

    # policy function
    def policy(self, state):
        x = self.m(self.pi_1(state))
        x = self.m(self.pi_2(x))
        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return Normal(mean, log_std.exp())

    # get sample probability
    def get_log_prob(self, state, action):
        normal = self.policy(state)
        return normal.log_prob(action)

    # sample from policy
    def forward(self, state):
        return self.policy(state).rsample()


# Gaussian RNN policy
class GaussianRNNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(GaussianRNNPolicy, self).__init__()

        # recursive network (we found GRU to be most effective, but this could
        # be replaced with RNN, LSTM, etc.)
        self.gru = nn.GRU(
                        input_size=state_dim,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True)

        # multi-layer perceptron
        self.pi_1 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # helper functions
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

    # policy function
    def policy(self, history):
        _, h_n = self.gru(history)
        x = self.m(self.pi_1(h_n.squeeze()))
        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return Normal(mean, log_std.exp())

    # get sample probability
    def get_log_prob(self, history, action):
        normal = self.policy(history)
        return normal.log_prob(action)

    # sample from policy
    def forward(self, history):
        return self.policy(history).rsample()        