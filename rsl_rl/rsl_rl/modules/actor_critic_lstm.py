import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# (以下 import 仅供参考，具体根据你项目的层次结构而定)
# from .actor_critic import ActorCritic  # rsl-rl 原有基类
# from .actor_critic import get_activation  # 如果你想复用 rsl-rl 的激活函数选择器
from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

# ----------------------------------------
# 1) Road Runner 中的若干辅助函数 (来自 base.py)
# ----------------------------------------
def normc_fn(m):
    """给 nn.Linear 做向量范数初始化."""
    if m.__class__.__name__.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def get_activation_rr(act_name: str):
    """
    Road Runner 里的激活函数选择。
    如果想跟 rsl-rl 的 get_activation 整合，可直接在 ActorCriticLSTM 里用 rsl-rl 的。
    这里仅作演示。
    """
    if act_name == 'relu':
        return F.relu
    elif act_name == 'tanh':
        return torch.tanh
    elif act_name == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError(f"Unsupported activation: {act_name}")

def create_lstm_layers(input_dim, layers):
    """
    Road runner 里 LSTMBase_ 使用过的创建多层 LSTMCell。  
    但官方 LSTMBase 使用的是 `nn.LSTM`。这里我们直接用 `nn.LSTM(...)` 即可。
    """
    # 如果你想自定义多层 LSTMCell，可以在此写。这里简单返回 nn.LSTM
    num_layers = len(layers)
    if num_layers < 1:
        raise ValueError("layers 不能为空，至少要有一个隐藏尺寸")
    hidden_size = layers[0]
    for h in layers:
        if h != hidden_size:
            raise ValueError("本示例仅示范多层相同 hidden_size，如 [128, 128, 128]")
    return nn.LSTM(input_dim, hidden_size, num_layers=num_layers)


# ----------------------------------------
# 2) Road Runner 的 LSTMBase (来自 base.py, 有删减/调整)
# ----------------------------------------
class LSTMBase(nn.Module):
    """
    与 road runner 中的 LSTMBase 类似，用 nn.LSTM 做时序模型。
    """
    def __init__(self, in_dim, layers):
        super().__init__()
        self.in_dim = in_dim
        self.layers = layers  # e.g. [128, 128]
        # 这里直接用多层 nn.LSTM
        self.lstm = create_lstm_layers(in_dim, layers)

        self.is_recurrent = True
        # 缓存当前隐藏状态 (h, c)
        self.hx = None

        # 下面三项是 road runner 用于输入归一化的统计量，这里可选
        self.welford_state_mean = torch.zeros(in_dim)
        self.welford_state_mean_diff = torch.ones(in_dim)
        self.welford_state_n = 1

        self.initialize_parameters()

    def initialize_parameters(self):
        # 给所有线性层做 normc 初始化
        self.apply(normc_fn)

    def init_hidden_state(self, batch_size=1, device='cpu'):
        """初始化隐藏状态 (h, c)."""
        num_layers = len(self.layers)
        hidden_size = self.layers[0]
        h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        self.hx = (h, c)

    def set_hidden_state(self, hx):
        """外部设置 (h, c). hx 应当是 (h, c) 组成的元组."""
        self.hx = hx

    def get_hidden_state(self):
        return self.hx

    def reset_hidden_state(self, dones=None):
        if self.hx is None:
            return

        h, c = self.hx  # h, c shape: (num_layers, batch_size, hidden_dim)
        # print("h size: ",h.size())
        # print("c size: ",c.size())
        if dones is None:
            # 如果 dones=None, 表示要重置整批
            h.zero_()
            c.zero_()
        else:
            # 假设 dones 是 (batch_size,) 或 (batch_size, 1)
            # 先找出所有 done==True 的环境索引
            done_indices = torch.nonzero(dones, as_tuple=False).flatten()
            if len(done_indices) > 0:
                # 对所有已经结束的环境对应的 hidden state 置零
                h[..., done_indices, :] = 0.0
                c[..., done_indices, :] = 0.0

        self.hx = (h, c)

        self.hx = (h, c)

    def normalize_state(self, state: torch.Tensor, update_normalization_param=True):
        """
        Road Runner 的在线归一化 (Welford)，仅用于演示，可按需关闭。
        仅在 dim = (features,) 的单步输入时更新统计。
        """
        if self.welford_state_n == 1:
            # 第一次初始化
            device = state.device
            self.welford_state_mean = torch.zeros(state.size(-1), device=device)
            self.welford_state_mean_diff = torch.ones(state.size(-1), device=device)

        if update_normalization_param and len(state.shape) == 1:
            old_mean = self.welford_state_mean.clone()
            self.welford_state_mean += (state - old_mean) / self.welford_state_n
            self.welford_state_mean_diff += (state - old_mean) * (state - self.welford_state_mean)
            self.welford_state_n += 1

        denom = torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)
        return (state - self.welford_state_mean) / denom

    def _base_forward(self, x: torch.Tensor):
        """
        执行 LSTM 前向。若 x 是 (batch_size, in_dim)，则在前面加 seq_len=1。
        """
        if len(x.shape) == 2:
            # (batch_size, in_dim) -> (1, batch_size, in_dim)
            x = x.unsqueeze(0)

        out, self.hx = self.lstm(x, self.hx)
        # out 形状: (seq_len, batch_size, hidden_size)
        # 取最后一个时刻 -> (batch_size, hidden_size)
        return out[-1]


# ----------------------------------------
# 3) Road Runner 的 Actor 和 Critic 基类 (actor.py, critic.py)
# ----------------------------------------
class Actor:
    """
    Road runner 中的 Actor 基类，负责最终输出动作 (mean, std)。
    """
    def __init__(self,
                 latent: int,       # 最后一层隐层的大小
                 action_dim: int,   # 动作维度
                 bounded: bool,     # 是否使用 tanh 限幅
                 learn_std: bool,   # 是否可学习 std
                 std: float):       # 若非 learn_std，则使用固定 std
        self.action_dim = action_dim
        self.bounded = bounded
        self.std_value = std  # 初始 std
        self.learn_std = learn_std

        self.means = nn.Linear(latent, action_dim)
        if self.learn_std:
            self.log_stds = nn.Linear(latent, action_dim)

    def _get_distribution_params(self, input_state, update_normalization_param):
        """
        执行 MLP/LSTM 后的输出 -> mean & std
        """
        state = self.normalize_state(input_state,
                                     update_normalization_param=update_normalization_param)
        latent = self._base_forward(state)  # 由子类去实现 base_forward
        mu = self.means(latent)
        if self.learn_std:
            # clamp 到 [-3, 0.5]，再 exponent
            std = torch.clamp(self.log_stds(latent), -3, 0.5).exp()
        else:
            std = torch.tensor(self.std_value, device=mu.device)
        return mu, std

    def pdf(self, state):
        """返回 Diagonal Normal 分布对象。"""
        mu, sd = self._get_distribution_params(state, update_normalization_param=False)
        return Normal(mu, sd)

    def log_prob(self, state, action):
        """给定 state 与 action，计算对数概率。"""
        dist = self.pdf(state)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        if self.bounded:
            # SAC, Appendix C trick => 减去 log(1 - tanh(a)^2)
            log_prob -= torch.log((1 - torch.tanh(action).pow(2)) + 1e-6).sum(-1, keepdim=True)
        return log_prob

    def actor_forward(self, state: torch.Tensor,
                      deterministic=True,
                      update_normalization_param=False,
                      return_log_prob=False):
        """Actor 前向，训练时可随机采样，推理时可用均值。"""
        mu, std = self._get_distribution_params(state, update_normalization_param)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        if self.bounded:
            action = torch.tanh(action)

        if return_log_prob:
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(action).pow(2)) + 1e-6).sum(-1, keepdim=True)
            return action, log_prob
        else:
            return action

class Critic:
    """
    Road runner 中的 Critic 基类。
    """
    def __init__(self, latent: int):
        self.critic_last_layer = nn.Linear(latent, 1)

    def critic_forward(self, state, update_norm=False):
        """输出 state 的价值 V(s)。"""
        state = self.normalize_state(state, update_normalization_param=update_norm)
        x = self._base_forward(state)
        return self.critic_last_layer(x)


# ----------------------------------------
# 4) Road Runner 的 LSTMActor, LSTMCritic
# ----------------------------------------
class LSTMActor(LSTMBase, Actor):
    def __init__(self,
                 obs_dim,            # 输入观测维度
                 action_dim,
                 layers,             # LSTM各层隐藏单元数, e.g. [128, 128]
                 bounded=True,
                 learn_std=True,
                 std=0.1):
        LSTMBase.__init__(self, in_dim=obs_dim, layers=layers)
        Actor.__init__(self,
                       latent=layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std)

    def forward(self, x, deterministic=True, update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x,
                                  deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

class LSTMCritic(LSTMBase, Critic):
    def __init__(self,
                 input_dim,
                 layers):
        LSTMBase.__init__(self, in_dim=input_dim, layers=layers)
        Critic.__init__(self, latent=layers[-1])

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)


# ----------------------------------------
# 5) 最终的 ActorCriticLSTM (与 ActorCriticRecurrent 风格/接口保持一致)
# ----------------------------------------
class ActorCriticLSTM(ActorCritic):
    """
    使用 Road Runner 完整的 LSTMActor + LSTMCritic 结构，
    但在对外接口上模仿 rsl-rl 的 ActorCriticRecurrent：

    - is_recurrent = True
    - reset(dones=None)
    - act(obs, masks=None, hidden_states=None)
    - act_inference(obs)
    - evaluate(critic_obs, masks=None, hidden_states=None)
    - get_hidden_states()
    """
    is_recurrent = True

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_layers=[128, 128],
                 critic_layers=[128, 128],
                 bounded=True,
                 learn_std=True,
                 init_noise_std=0.1,
                 activation='relu',
                 **kwargs):
        """
        这里把 road runner 的超参数做一个映射：
        - actor_layers => LSTM 的 hidden_size 列表
        - critic_layers => LSTM 的 hidden_size 列表
        - bounded, learn_std => 是否做 tanh & 是否学习 std
        - init_noise_std => 初始常数 std
        - activation => 主要会作用在 LSTM 后面的激活(若需要)
        - ...
        """
        if kwargs:
            print("ActorCriticLSTM got extra kwargs:", kwargs.keys())
        super().__init__(num_actor_obs=num_actor_obs,
                         num_critic_obs=num_critic_obs,
                         num_actions=num_actions,
                         actor_hidden_dims=[1],  # 占位，不用 rsl-rl 父类的 MLP
                         critic_hidden_dims=[1], # 占位
                         activation=activation,
                         init_noise_std=init_noise_std)

        # 构造 Road Runner 风格的 LSTM Actor & Critic
        self.actor = LSTMActor(
            obs_dim=num_actor_obs,
            action_dim=num_actions,
            layers=actor_layers,
            bounded=bounded,
            learn_std=learn_std,
            std=init_noise_std
        )
        self.critic = LSTMCritic(
            input_dim=num_critic_obs,
            layers=critic_layers
        )

        # 让我们看看是否需要 print
        print(f"Road Runner LSTM Actor: {self.actor}")
        print(f"Road Runner LSTM Critic: {self.critic}")

        # 我们可以用 self.distribution 记录上一次 forward 的分布(便于 rsl-rl 算log_prob/entropy)
        self.distribution = None

    def reset(self, dones=None):
        """
        在 RNN 场景下，每个 rollout 开始或在 dones=True 时，对相应环境的隐藏状态清零。
        如果 dones is None，表示全部重置。
        如果 dones 里是 [True, False, True...]，则只对 True 的索引重置。
        """
        # print(dones.size())
        self.actor.reset_hidden_state(dones)
        self.critic.reset_hidden_state(dones)

    def get_hidden_states(self):
        """
        返回 (actor_h, actor_c), (critic_h, critic_c)，
        以便在 rollout 时保存 / 后续 set_hidden_states() 中恢复。
        """
        return self.actor.get_hidden_state(), self.critic.get_hidden_state()

    def set_hidden_states(self, actor_hx, critic_hx):
        """
        从外部恢复隐藏状态；(actor_hx, critic_hx) 都是 (h, c) 形式。
        """
        self.actor.set_hidden_state(actor_hx)
        self.critic.set_hidden_state(critic_hx)

    def act(self, observations, masks=None, hidden_states=None):
        """
        与 rsl-rl 的 ActorCriticRecurrent 类似签名：
        - observations: (batch_size, obs_dim) or (obs_dim,)
        - masks / hidden_states: 可选，用于在 policy 更新时做序列展开。
        
        这里演示：
          - 若 hidden_states 不为空，则先 set_hidden_states
          - 若 masks 不为空，则做 partial reset (可选)
          - 最后调用 self.actor(...) 获取动作分布并采样
        """
        if hidden_states is not None:
            # hidden_states[0] 是 actor_hx, hidden_states[1] 是 critic_hx
            self.set_hidden_states(hidden_states[0], hidden_states[1])

        if masks is not None:
            # 若你想根据 masks 重置某些环境的 hidden，可在此做:
            # 例如 (masks == 0)->done
            dones = (masks == 0).squeeze(-1)  # 具体看你数据维度
            print("dones: ",dones.size())
            self.actor.reset_hidden_state()
            self.critic.reset_hidden_state(dones)
            observations = unpad_trajectories(observations, masks)
        print("observations: ",observations.size())
        # Road runner 中获得分布:
        dist = self.actor.pdf(observations)
        action = dist.sample()
        if self.actor.bounded:
            action = torch.tanh(action)

        self.distribution = dist
        return action

    def act_inference(self, observations):
        """
        测试/推理时使用的纯策略均值 (deterministic).
        """
        # 直接拿均值:
        dist = self.actor.pdf(observations)
        mean = dist.mean
        if self.actor.bounded:
            mean = torch.tanh(mean)

        return mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        计算 V(s). 同理可处理 masks/hidden_states。
        """
        if hidden_states is not None:
            self.set_hidden_states(hidden_states[0], hidden_states[1])
        if masks is not None:
            dones = (masks == 0).squeeze(-1)
            self.actor.reset_hidden_state(dones)
            self.critic.reset_hidden_state(dones)

        return self.critic(critic_observations)

    # rsl-rl 常见的一些接口：
    @property
    def entropy(self):
        """本次分布的熵。"""
        if self.distribution is None:
            return None
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        """给定动作计算 log_prob（sum over action dim）。"""
        if self.distribution is None:
            return None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_distribution(self, observations):
        """
        若在 rsl-rl 训练环节需要先 update_distribution(observations) 再做 get_actions_log_prob，
        可以用这个函数。它会把分布缓存在 self.distribution。
        """
        dist = self.actor.pdf(observations)
        self.distribution = dist
