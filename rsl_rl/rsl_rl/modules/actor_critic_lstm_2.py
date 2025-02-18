import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import split_and_pad_trajectories,unpad_trajectories
class ActorCriticLSTM(nn.Module):
    """ Recurrent policy using LSTM cells, compatible with rsl-rl's storage and training pipeline """
    is_recurrent = True
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()
        
        # Actor Network
        self.actor_rnn = LSTMModule(num_actor_obs, rnn_hidden_size, rnn_num_layers)
        self.actor_mlp = MLP(rnn_hidden_size, num_actions, actor_hidden_dims, activation)
        
        # Critic Network  
        self.critic_rnn = LSTMModule(num_critic_obs, rnn_hidden_size, rnn_num_layers)
        self.critic_mlp = MLP(rnn_hidden_size, 1, critic_hidden_dims, activation)
        
        print(f"Actor RNN: {self.actor_rnn}")
        print(f"Critic RNN: {self.critic_rnn}")
        print(f"Actor MLP: {self.actor_mlp}")
        print(f"Critic MLP: {self.critic_mlp}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None): # ok
        self.actor_rnn.reset(dones)
        self.critic_rnn.reset(dones)

    def update_distribution(self, observations):
        actor_features = self.actor_rnn(observations)
        actions_mean = self.actor_mlp(actor_features.squeeze(0))# ok
        self.distribution = Normal(actions_mean, actions_mean*0. + self.std)

    def act(self, observations, masks=None, hidden_states=None): # no ok
        # Process temporal seque               
        actor_features = self.actor_rnn(observations,masks,hidden_states)
        # Feed through MLP      
        actions_mean = self.actor_mlp(actor_features.squeeze(0))# ok
        self.distribution = Normal(actions_mean, self.std)# ok
        
        return self.distribution.sample()# ok
    
    def act_inference(self, observations):
        actions_mean = self.actor_rnn(observations)
        return self.actor_mlp(actions_mean.squeeze(0))
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):# ok
        # Process temporal sequence
        critic_features = self.critic_rnn(critic_observations, masks, hidden_states) # no ok
        
        # Feed through MLP
        return self.critic_mlp(critic_features.squeeze(0))# ok

    def get_actions_log_prob(self, actions):# ok
        return self.distribution.log_prob(actions).sum(-1)# ok
    
    @property
    def entropy(self):# ok
        return self.distribution.entropy().sum(dim=-1)# ok
    
    @property
    def action_std(self):# ok
        return self.distribution.stddev# ok

    @property
    def action_mean(self):# ok
        return self.distribution.mean# ok

    def get_hidden_states(self):
        return (self.actor_rnn.hidden_states, 
                self.critic_rnn.hidden_states)

class LSTMModule(nn.Module):
    """ LSTM module compatible with rsl-rl's hidden state management """
    def __init__(self, input_size, hidden_size = 256, num_layers = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.lstm(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.lstm(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0

class MLP(nn.Module):
    """ Basic MLP network with activation """
    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev_dim, dim), get_activation(activation)]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
