"""
MAPPO (Multi-Agent PPO) - Fully aligned with the original cMALC-D project implementation
Key features:
1. Uses n-step returns (instead of GAE)
2. Target Critic network
3. Old Policy network (for importance sampling)
4. Running Mean/Std normalization
5. Mask mechanism to handle variable-length episodes
6. Agent ID and Last Action as inputs
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from .running_mean_std import RunningMeanStd

# ============== 网络架构 ==============
class RNNAgent(nn.Module):
    """Actor network - aligned with the original rnn_agent.py"""
    def __init__(self, input_shape, n_actions, hidden_dim=128, use_rnn=True):
        super(RNNAgent, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        if use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
    
    def init_hidden(self):
        """Initialize hidden state"""
        return self.fc1.weight.new(1, self.hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        """
        Args:
            inputs: (bs * n_agents, input_shape)
            hidden_state: (bs * n_agents, hidden_dim)
        Returns:
            q: (bs * n_agents, n_actions)
            h: (bs * n_agents, hidden_dim)
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        if self.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

class CentralVCritic(nn.Module):
    """Centralized value function - aligned with the original centralV.py"""
    def __init__(self, state_shape, n_agents, hidden_dim=128, 
                 obs_agent_id=True, obs_individual_obs=False):
        super(CentralVCritic, self).__init__()
        self.n_agents = n_agents
        self.obs_agent_id = obs_agent_id
        self.obs_individual_obs = obs_individual_obs
        
        # 计算输入维度：state + agent_id
        input_shape = state_shape
        if obs_agent_id:
            input_shape += n_agents
        
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, agent_ids=None):
        """
        Args:
            state: (bs, T, n_agents, state_dim) or (bs, T, 1, state_dim)
            agent_ids: (bs, T, n_agents, n_agents)
        Returns:
            values: (bs, T, n_agents, 1)
        """
        x = state
        if self.obs_agent_id and agent_ids is not None:
            x = torch.cat([x, agent_ids], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

# ============== Multi-Agent Controller (MAC) ==============
class BasicMAC:
    """
    Multi-Agent Controller - manages all agents' policies
    Aligned with the original basic_controller.py
    """
    def __init__(self, n_agents, obs_shape, n_actions, args):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.args = args
        
        # 计算输入维度
        input_shape = obs_shape
        if args.get('obs_last_action', False):
            input_shape += n_actions
        if args.get('obs_agent_id', True):
            input_shape += n_agents
        
        # 创建agent网络（所有智能体共享参数）
        self.agent = RNNAgent(
            input_shape, 
            n_actions, 
            args.get('hidden_dim', 128),
            args.get('use_rnn', True)
        )
        
        self.hidden_states = None
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for all agents"""
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )

    def forward(self, batch_obs, batch_last_actions=None, test_mode=False):
        """
        Forward pass
        Args:
            batch_obs: (bs, n_agents, obs_dim)
            batch_last_actions: (bs, n_agents, n_actions) one-hot
        Returns:
            agent_outs: (bs, n_agents, n_actions) - policy logits/softmax
        """
        bs = batch_obs.shape[0]
        
        # 构建输入
        inputs = [batch_obs]
        
        if self.args.get('obs_last_action', False) and batch_last_actions is not None:
            inputs.append(batch_last_actions)
        
        if self.args.get('obs_agent_id', True):
            agent_ids = torch.eye(self.n_agents, device=batch_obs.device).unsqueeze(0).expand(bs, -1, -1)
            inputs.append(agent_ids)
        
        # 拼接并reshape
        inputs = torch.cat(inputs, dim=-1)  # (bs, n_agents, input_dim)
        inputs = inputs.reshape(bs * self.n_agents, -1)  # (bs*n_agents, input_dim)
        
        # 通过网络
        agent_outs, self.hidden_states = self.agent(inputs, self.hidden_states.reshape(-1, self.agent.hidden_dim))
        
        # Reshape回来
        agent_outs = agent_outs.view(bs, self.n_agents, -1)  # (bs, n_agents, n_actions)
        self.hidden_states = self.hidden_states.view(bs, self.n_agents, -1)
        
        # Softmax输出策略概率
        if self.args.get('agent_output_type', 'pi_logits') == 'pi_logits':
            agent_outs = F.softmax(agent_outs, dim=-1)
        
        return agent_outs
    
    def parameters(self):
        return self.agent.parameters()
    
    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
    
    def state_dict(self):
        return self.agent.state_dict()
    
    def load_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)
    
# ============== Episode Buffer ==============
class EpisodeBuffer:
    """
    Stores all transitions of a single episode
    Aligned with the original project data structure
    """
    def __init__(self, n_agents, obs_shape, n_actions, max_seq_length=1000):
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.max_seq_length = max_seq_length
        
        self.reset()
    
    def reset(self):
        """Reset the buffer"""
        self.data = {
            'obs': [],           # (T, n_agents, obs_dim)
            'actions': [],       # (T, n_agents)
            'actions_onehot': [],  # (T, n_agents, n_actions)
            'reward': [],        # (T, n_agents) or (T, 1)
            'state': [],         # (T, state_dim) - 全局状态
            'terminated': [],    # (T,)
            'filled': [],        # (T,)
        }
        self.t = 0
    
    def add(self, obs, actions, reward, state, terminated):
        """
        Add a transition
        """
        # 转换actions为one-hot
        actions_onehot = F.one_hot(
            torch.LongTensor(actions), 
            num_classes=self.n_actions
        ).float().numpy()
        
        self.data['obs'].append(obs)
        self.data['actions'].append(actions)
        self.data['actions_onehot'].append(actions_onehot)
        self.data['reward'].append(reward)
        self.data['state'].append(state)
        self.data['terminated'].append(terminated)
        self.data['filled'].append(1.0)
        
        self.t += 1
    
    def get_batch(self):
        """
        Get batch data (adds batch dimension)
        Returns: dict with tensors of shape (1, T+1, ...)
        """
        batch = {}
        
        # 添加最后一个dummy transition（用于bootstrap）
        T = len(self.data['obs'])
        
        for key in ['obs', 'actions', 'actions_onehot', 'reward', 'state']:
            # 复制最后一帧
            data_list = self.data[key] + [self.data[key][-1]]
            batch[key] = torch.FloatTensor(np.array(data_list)).unsqueeze(0)  # (1, T+1, ...)
        
        # terminated和filled
        batch['terminated'] = torch.FloatTensor(
            self.data['terminated'] + [1.0]
        ).unsqueeze(0).unsqueeze(-1)  # (1, T+1, 1)
        
        batch['filled'] = torch.FloatTensor(
            self.data['filled'] + [0.0]
        ).unsqueeze(0).unsqueeze(-1)  # (1, T+1, 1)
        
        # 将actions转换为long
        batch['actions'] = batch['actions'].long().unsqueeze(-1)  # (1, T+1, n_agents, 1)
        
        batch['batch_size'] = 1
        batch['max_seq_length'] = T + 1
        
        return batch
    
    def __len__(self):
        return self.t

# ============== MAPPO Learner ==============
class MAPPOLearner:
    """
    MAPPO trainer - fully aligned with the original ppo_learner.py
    """
    def __init__(self, mac, n_agents, obs_shape, n_actions, args):
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.mac = mac
        
        # 创建旧策略（用于重要性采样）
        self.old_mac = copy.deepcopy(mac)
        
        # Actor优化器
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args['lr'])
        
        # 创建Critic
        # state_dim = obs_dim * n_agents (拼接所有观测)
        state_dim = obs_shape * n_agents
        self.critic = CentralVCritic(
            state_dim, 
            n_agents, 
            args.get('hidden_dim', 128),
            obs_agent_id=args.get('obs_agent_id', True)
        )
        self.target_critic = copy.deepcopy(self.critic)
        
        # Critic优化器
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args['lr'])
        
        # 标准化
        device = args.get('device', 'cpu')
        if args.get('standardise_returns', False):
            self.ret_ms = RunningMeanStd(shape=(n_agents,), device=device)
        if args.get('standardise_rewards', True):
            rew_shape = (1,) if args.get('common_reward', False) else (n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)
        self.last_target_update_step = 0
        self.critic_training_steps = 0
    
    def train(self, batch, device='cpu'):
        """
        Train on a batch
        Args:
            batch: dict from EpisodeBuffer.get_batch()
        Returns:
            log_stats: dict of training statistics
        """
        # 移动到device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # 提取数据
        rewards = batch["reward"][:, :-1]  # (bs, T, n_agents)
        actions = batch["actions"][:, :-1]  # (bs, T, n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()  # (bs, T, 1)
        mask = batch["filled"][:, :-1].float()  # (bs, T, 1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 标准化奖励
        if self.args.get('standardise_rewards', True):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var + 1e-8)
        
        # 扩展reward为每个agent
        if self.args.get('common_reward', False):
            rewards = rewards.expand(-1, -1, self.n_agents)
        
        mask = mask.repeat(1, 1, self.n_agents)  # (bs, T, n_agents)
        
        # ============== 计算old policy ==============
        bs = batch['batch_size']
        T = batch['max_seq_length'] - 1

        old_mac_out = []
        self.old_mac.init_hidden(bs)
        for t in range(T):
            obs_t = batch['obs'][:, t]  # (bs, n_agents, obs_dim)
            last_actions_t = batch['actions_onehot'][:, t-1] if t > 0 else torch.zeros_like(batch['actions_onehot'][:, 0])
            agent_outs = self.old_mac.forward(obs_t, last_actions_t)
            old_mac_out.append(agent_outs)
        old_mac_out = torch.stack(old_mac_out, dim=1)  # (bs, T, n_agents, n_actions)
        
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0
        old_pi_taken = torch.gather(old_pi, dim=3, index=actions).squeeze(3)  # (bs, T, n_agents)
        old_log_pi_taken = torch.log(old_pi_taken + 1e-10)
        
        # ============== 多轮epoch训练 ==============
        for k in range(self.args.get('epochs', 4)):
            # 计算当前policy
            mac_out = []
            self.mac.init_hidden(bs)
            for t in range(T):
                obs_t = batch['obs'][:, t]
                last_actions_t = batch['actions_onehot'][:, t-1] if t > 0 else torch.zeros_like(batch['actions_onehot'][:, 0])
                agent_outs = self.mac.forward(obs_t, last_actions_t)
                mac_out.append(agent_outs)
            mac_out = torch.stack(mac_out, dim=1)  # (bs, T, n_agents, n_actions)
            
            pi = mac_out
            
            # ============== 训练Critic ==============
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, mask
            )
            advantages = advantages.detach()

            # ============== 训练Actor ==============
            pi[mask == 0] = 1.0
            pi_taken = torch.gather(pi, dim=3, index=actions).squeeze(3)  # (bs, T, n_agents)
            log_pi_taken = torch.log(pi_taken + 1e-10)
            
            # PPO clip
            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 
                               1 - self.args.get('eps_clip', 0.2), 
                               1 + self.args.get('eps_clip', 0.2)) * advantages
            
            # 熵
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)  # (bs, T, n_agents)
            
            # Actor loss
            pg_loss = -(
                (torch.min(surr1, surr2) + self.args.get('entropy_coef', 0.01) * entropy) * mask
            ).sum() / mask.sum()
            
            # 更新Actor
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.get('max_grad_norm', 10.0)
            )
            self.agent_optimiser.step()
        
        # 复制当前策略到old_mac
        self.old_mac.load_state(self.mac)

        # 更新target critic
        self.critic_training_steps += 1
        target_update_interval = self.args.get('target_update_interval_or_tau', 0.01)
        if target_update_interval > 1:
            # 硬更新
            if (self.critic_training_steps - self.last_target_update_step) >= target_update_interval:
                self._update_targets_hard()
                self.last_target_update_step = self.critic_training_steps
        else:
            # 软更新
            self._update_targets_soft(target_update_interval)
        
        # 统计
        log_stats = {}
        for key in critic_train_stats:
            log_stats[key] = critic_train_stats[key]
        log_stats["advantage_mean"] = (advantages * mask).sum().item() / mask.sum().item()
        log_stats["pg_loss"] = pg_loss.item()
        log_stats["agent_grad_norm"] = grad_norm.item()
        log_stats["pi_max"] = (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item()
        
        return log_stats
    
    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        """Train Critic (aligned with original project)"""
        bs = batch['batch_size']
        T = batch['max_seq_length'] - 1
        
        # 构建critic输入
        state = batch['state'][:, :-1]  # (bs, T, state_dim)
        state = state.unsqueeze(2).repeat(1, 1, self.n_agents, 1)  # (bs, T, n_agents, state_dim)
        
        agent_ids = torch.eye(self.n_agents, device=state.device).unsqueeze(0).unsqueeze(0).expand(bs, T, -1, -1)
        
        # 计算target values
        with torch.no_grad():
            state_next = batch['state'][:, 1:]  # (bs, T, state_dim)
            state_next = state_next.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            agent_ids_next = torch.eye(self.n_agents, device=state.device).unsqueeze(0).unsqueeze(0).expand(bs, T, -1, -1)
            target_vals = target_critic(state_next, agent_ids_next).squeeze(3)  # (bs, T, n_agents)
        
        if self.args.get('standardise_returns', False):
            target_vals = target_vals * torch.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean
        
        # N-step returns
        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.get('q_nstep', 5)
        )
        
        if self.args.get('standardise_returns', False):
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)
        
        # 当前values
        v = critic(state, agent_ids).squeeze(3)  # (bs, T, n_agents)
        
        # TD error
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask

        # Critic loss
        loss = (masked_td_error ** 2).sum() / mask.sum()
        
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.get('max_grad_norm', 10.0)
        )
        self.critic_optimiser.step()
        
        # 统计
        running_log = {
            "critic_loss": loss.item(),
            "critic_grad_norm": grad_norm.item(),
            "td_error_abs": (masked_td_error.abs().sum() / mask.sum()).item(),
            "q_taken_mean": (v * mask).sum().item() / mask.sum().item(),
            "target_mean": (target_returns * mask).sum().item() / mask.sum().item(),
        }
        
        return masked_td_error, running_log
    
    def nstep_returns(self, rewards, mask, values, nsteps):
        """
        Compute n-step returns
        Aligned with original project implementation
        """
        nstep_values = torch.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = torch.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args['gamma'] ** step * values[:, t] * mask[:, t]
                    )
                else:
                    nstep_return_t += (
                        self.args['gamma'] ** step * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values
    
    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# ============== MAPPO Agent (接口) ==============
class MAPPOAgent:
    """
    MAPPO Agent - fully aligned with the original cMALC-D project
    Provides the interface to interact with the environment
    """
    def __init__(self, obs_dim, act_dim, n_agents,
                 lr=3e-4, gamma=0.99, eps_clip=0.2,
                 epochs=4, hidden_dim=128, use_rnn=True,
                 entropy_coef=0.01, q_nstep=5,
                 max_grad_norm=10.0, device='cpu',
                 standardise_rewards=True, standardise_returns=False,
                 obs_agent_id=True, obs_last_action=False,
                 target_update_interval_or_tau=0.01,
                 common_reward=True):
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        
        # 配置
        self.args = {
            'lr': lr,
            'gamma': gamma,
            'eps_clip': eps_clip,
            'epochs': epochs,
            'hidden_dim': hidden_dim,
            'use_rnn': use_rnn,
            'entropy_coef': entropy_coef,
            'q_nstep': q_nstep,
            'max_grad_norm': max_grad_norm,
            'device': device,
            'standardise_rewards': standardise_rewards,
            'standardise_returns': standardise_returns,
            'obs_agent_id': obs_agent_id,
            'obs_last_action': obs_last_action,
            'agent_output_type': 'pi_logits',
            'common_reward': common_reward,  # 每个agent有独立奖励
            'target_update_interval_or_tau': target_update_interval_or_tau,
        }

        # 创建MAC和Learner
        self.mac = BasicMAC(n_agents, obs_dim, act_dim, self.args)
        self.mac.agent.to(self.device)
        
        self.learner = MAPPOLearner(
            self.mac, n_agents, obs_dim, act_dim, self.args
        )
        self.learner.critic.to(self.device)
        self.learner.target_critic.to(self.device)
        
        # Episode buffer
        self.buffer = EpisodeBuffer(n_agents, obs_dim, act_dim)
        
        # Last actions (用于下一步输入)
        self.last_actions = None
    
    def reset(self):
        """Reset (call at the start of an episode)"""
        self.mac.init_hidden(batch_size=1)
        self.buffer.reset()
        self.last_actions = None
    
    def select_action(self, obs_list, deterministic=False):
        """
        Select actions
        Args:
            obs_list: list of observations, len=n_agents
            deterministic: whether to choose deterministically
        Returns:
            actions: (n_agents,) numpy array
        """
        with torch.no_grad():
            # 转换为tensor (1, n_agents, obs_dim)
            obs = torch.FloatTensor(np.array(obs_list)).unsqueeze(0).to(self.device)
            
            # 前向传播
            pi = self.mac.forward(obs, self.last_actions)  # (1, n_agents, n_actions)
            pi = pi.squeeze(0)  # (n_agents, n_actions)
            
            # 采样动作
            if deterministic:
                actions = torch.argmax(pi, dim=-1)
            else:
                dist = Categorical(probs=pi)
                actions = dist.sample()
            
            actions_np = actions.cpu().numpy()
            
            # 保存last_actions（one-hot）
            self.last_actions = F.one_hot(actions, num_classes=self.act_dim).float().unsqueeze(0).to(self.device)
            
            return actions_np
    
    def store_transition(self, obs_list, actions, rewards):
        """
        Store a transition
        Args:
            obs_list: list of observations
            actions: (n_agents,) numpy array
            rewards: (n_agents,) numpy array
        """
        obs = np.array(obs_list)  # (n_agents, obs_dim)
        state = obs.flatten()  # (obs_dim * n_agents,) 全局状态
        # ✅ 如果是common_reward，只存储一个标量值
        if self.args['common_reward']:
            # 取第一个agent的奖励（因为所有agent奖励相同）
            reward = np.array([rewards[0]])  # (1,)
        else:
            reward = rewards  # (n_agents,)
        self.buffer.add(
            obs=obs,
            actions=actions,
            reward=rewards,
            state=state,
            terminated=False
        )
    
    def finish_episode(self, final_obs_list):
        """
        Episode ended, mark the last transition
        """
        if len(self.buffer) > 0:
            # 标记最后一个为terminated
            self.buffer.data['terminated'][-1] = True
    
    def update(self):
        """
        Update networks (call after episode ends)
        """
        if len(self.buffer) == 0:
            return {}
        
        # 获取batch
        batch = self.buffer.get_batch()
        
        # 训练
        log_stats = self.learner.train(batch, device=self.device)
        
        # 清空buffer
        self.buffer.reset()
        
        return log_stats
    
    def save(self, path):
        """Save model (including networks, optimizers, and normalization stats)"""
        save_dict = {
            'mac': self.mac.state_dict(),
            'critic': self.learner.critic.state_dict(),
            'actor_optimizer': self.learner.agent_optimiser.state_dict(),
            'critic_optimizer': self.learner.critic_optimiser.state_dict(),
        }
        
        # 保存 RunningMeanStd 的状态（奖励标准化）
        if hasattr(self.learner, 'rew_ms'):
            save_dict['rew_ms'] = {
                'mean': self.learner.rew_ms.mean,
                'var': self.learner.rew_ms.var,
                'count': self.learner.rew_ms.count
            }
        
        # 保存 RunningMeanStd 的状态（回报标准化）
        if hasattr(self.learner, 'ret_ms'):
            save_dict['ret_ms'] = {
                'mean': self.learner.ret_ms.mean,
                'var': self.learner.ret_ms.var,
                'count': self.learner.ret_ms.count
            }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model (including networks, optimizers, and normalization stats)"""
        checkpoint = torch.load(path, map_location=self.device)
        self.mac.load_state_dict(checkpoint['mac'])
        self.learner.critic.load_state_dict(checkpoint['critic'])
        self.learner.agent_optimiser.load_state_dict(checkpoint['actor_optimizer'])
        self.learner.critic_optimiser.load_state_dict(checkpoint['critic_optimizer'])
        
        # 恢复 RunningMeanStd 的状态（奖励标准化）
        if 'rew_ms' in checkpoint and hasattr(self.learner, 'rew_ms'):
            self.learner.rew_ms.mean = checkpoint['rew_ms']['mean'].to(self.device)
            self.learner.rew_ms.var = checkpoint['rew_ms']['var'].to(self.device)
            self.learner.rew_ms.count = checkpoint['rew_ms']['count']
            print(f"   └─ Reward normalizer restored: count={self.learner.rew_ms.count:.0f}")
        
        # 恢复 RunningMeanStd 的状态（回报标准化）
        if 'ret_ms' in checkpoint and hasattr(self.learner, 'ret_ms'):
            self.learner.ret_ms.mean = checkpoint['ret_ms']['mean'].to(self.device)
            self.learner.ret_ms.var = checkpoint['ret_ms']['var'].to(self.device)
            self.learner.ret_ms.count = checkpoint['ret_ms']['count']
            print(f"   └─ Return normalizer restored: count={self.learner.ret_ms.count:.0f}")
        
        print(f"Model loaded from {path}")