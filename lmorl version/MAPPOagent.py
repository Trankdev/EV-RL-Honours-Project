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
    """Centralized value function - aligned with the original centralV.py

    NEW FOR LEXICOGRAPHIC MORL: outputs n_objectives values per agent instead
    of 1 (one value head per priority level, sharing the trunk). Legacy
    single-objective use just passes n_objectives=1 and everything behaves
    exactly as before.
    """
    def __init__(self, state_shape, n_agents, hidden_dim=128, 
                 obs_agent_id=True, obs_individual_obs=False, n_objectives=1):
        super(CentralVCritic, self).__init__()
        self.n_agents = n_agents
        self.obs_agent_id = obs_agent_id
        self.obs_individual_obs = obs_individual_obs
        self.n_objectives = n_objectives
        
        # 计算输入维度：state + agent_id
        input_shape = state_shape
        if obs_agent_id:
            input_shape += n_agents
        
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_objectives)
    
    def forward(self, state, agent_ids=None):
        """
        Args:
            state: (bs, T, n_agents, state_dim) or (bs, T, 1, state_dim)
            agent_ids: (bs, T, n_agents, n_agents)
        Returns:
            values: (bs, T, n_agents, n_objectives)
        """
        x = state
        if self.obs_agent_id and agent_ids is not None:
            x = torch.cat([x, agent_ids], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)  # (..., n_objectives)
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

    NEW FOR LEXICOGRAPHIC MORL: when args['n_objectives'] > 1 and
    args['lexicographic'] is True, this implements a practical Lagrangian
    simplification of Skalse et al. (IJCAI-22) Lexicographic PPO, adapted
    from their continuous online-update algorithm to this codebase's
    episodic batch training (one episode collected, then `epochs` PPO
    passes over it). See the long comment above `_lexicographic_actor_loss`
    for exactly what is simplified and why - this is NOT a literal
    reimplementation of their Algorithm 3, and should be described as an
    adaptation in any writeup.
    """
    def __init__(self, mac, n_agents, obs_shape, n_actions, args):
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.mac = mac

        self.n_objectives = args.get('n_objectives', 1)
        self.lexicographic = args.get('lexicographic', False) and self.n_objectives > 1
        # tolerance (tau_i): how far objective i is allowed to drop below its
        # locked-in reference level before its Lagrange multiplier kicks in.
        # One entry per CONSTRAINED objective, i.e. length n_objectives - 1
        # (the lowest-priority objective is never itself a constraint).
        lex_tol = args.get('lex_tolerance', 0.0)
        n_constraints = max(self.n_objectives - 1, 0)
        self.lex_tolerance = (list(lex_tol) if isinstance(lex_tol, (list, tuple))
                               else [lex_tol] * n_constraints)
        self.lex_dual_lr = args.get('lex_dual_lr', 0.05)      # eta: Lagrange multiplier learning rate
        self.lex_ema_rho = args.get('lex_ema_rho', 0.05)      # smoothing for the ratcheted threshold EMA
        # base_weight_i: fixed priority-decay so every objective still gets
        # *some* direct gradient even while its multiplier is 0 (keeps
        # higher-priority objectives improving early in training, before
        # there's anything for lambda to defend yet). Geometric decay by
        # priority index, index 0 = highest priority = biggest base weight.
        base_decay = args.get('lex_base_weight_decay', 0.1)
        self.lex_base_weights = [base_decay ** i for i in range(self.n_objectives)]

        # Persistent Lagrangian state (survives across episodes/updates).
        self.lex_lambda = [0.0 for _ in range(n_constraints)]
        self.lex_k_ema = [None for _ in range(n_constraints)]   # None = not yet observed

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
            obs_agent_id=args.get('obs_agent_id', True),
            n_objectives=self.n_objectives
        )
        self.target_critic = copy.deepcopy(self.critic)
        
        # Critic优化器
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args['lr'])
        
        # 标准化 - shape now carries the objective dimension too
        device = args.get('device', 'cpu')
        if args.get('standardise_returns', False):
            self.ret_ms = RunningMeanStd(shape=(n_agents, self.n_objectives), device=device)
        if args.get('standardise_rewards', True):
            self.rew_ms = RunningMeanStd(shape=(n_agents, self.n_objectives), device=device)
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
        # NEW FOR LEXICOGRAPHIC MORL: reward now carries a trailing objective
        # dimension always - (bs, T, n_agents, n_objectives). For legacy
        # single-objective runs n_objectives == 1 and this is unchanged
        # behaviour with one extra size-1 axis.
        rewards = batch["reward"][:, :-1]  # (bs, T, n_agents, n_objectives)
        actions = batch["actions"][:, :-1]  # (bs, T, n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()  # (bs, T, 1)
        mask = batch["filled"][:, :-1].float()  # (bs, T, 1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents)  # (bs, T, n_agents)

        # keep an UN-normalised copy per objective, used only to decide
        # whether a constrained objective was "active" this episode (e.g.
        # whether an EV appeared at all) - see _update_lexicographic_duals.
        raw_rewards = rewards.clone()

        # 标准化奖励 (per-objective, since objectives can live on very
        # different scales - e.g. EV timeloss vs. summed queue delay)
        if self.args.get('standardise_rewards', True):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var + 1e-8)

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

        target_returns = None
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
            # advantages, target_returns now (bs, T, n_agents, n_objectives)
            advantages, critic_train_stats, target_returns = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, mask
            )
            advantages = advantages.detach()

            # ============== 训练Actor ==============
            pi[mask == 0] = 1.0
            pi_taken = torch.gather(pi, dim=3, index=actions).squeeze(3)  # (bs, T, n_agents)
            log_pi_taken = torch.log(pi_taken + 1e-10)
            
            # PPO clip (same importance ratio for every objective - it's one
            # shared policy, only the advantage stream differs per-objective)
            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())  # (bs, T, n_agents)
            r = ratios.unsqueeze(-1)  # (bs, T, n_agents, 1) - broadcast over objectives
            surr1 = r * advantages
            surr2 = torch.clamp(r,
                               1 - self.args.get('eps_clip', 0.2),
                               1 + self.args.get('eps_clip', 0.2)) * advantages
            obj_surrogate = torch.min(surr1, surr2)  # (bs, T, n_agents, n_objectives)

            # 熵 (regularises the final combined policy, not any one objective)
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)  # (bs, T, n_agents)

            if self.lexicographic:
                actor_objective, lex_log = self._lexicographic_actor_loss(obj_surrogate, mask)
            else:
                # legacy path: objective 0 only (n_objectives==1 case is
                # numerically identical to the pre-LMORL code)
                actor_objective = obj_surrogate[..., 0]
                lex_log = {}

            # Actor loss
            pg_loss = -(
                (actor_objective + self.args.get('entropy_coef', 0.01) * entropy) * mask
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

        # NEW FOR LEXICOGRAPHIC MORL: update Lagrange multipliers / ratcheted
        # thresholds ONCE per train() call (i.e. once per episode), using
        # this episode's final n-step return estimates. This is the
        # "dual ascent" half of the Lagrangian relaxation - see the big
        # comment on _lexicographic_actor_loss for the full picture.
        if self.lexicographic:
            self._update_lexicographic_duals(target_returns, raw_rewards, mask)

        # 统计
        log_stats = {}
        for key in critic_train_stats:
            log_stats[key] = critic_train_stats[key]
        for i in range(self.n_objectives):
            log_stats[f"advantage_mean_r{i+1}"] = (
                (advantages[..., i] * mask).sum().item() / mask.sum().item()
            )
        log_stats.update(lex_log)
        log_stats["pg_loss"] = pg_loss.item()
        log_stats["agent_grad_norm"] = grad_norm.item()
        log_stats["pi_max"] = (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item()
        
        return log_stats

    # =====================================================================
    # ============== Lexicographic MORL: Lagrangian actor loss =============
    # =====================================================================
    # Adapted from Skalse et al. (IJCAI-22) "Lexicographic Multi-Objective
    # Reinforcement Learning", Section 3.2 (LPPO). Their Algorithm 3 assumes
    # continuous, per-timestep online updates across multiple asynchronous
    # timescales, and a formal "has objective i converged yet" test. This
    # codebase collects one full episode, then runs `epochs` PPO passes over
    # it, then discards the batch - there is no meaningful per-timestep
    # timescale separation to exploit, and no clean automatic convergence
    # test. The adaptation made here, once per train() call (= once per
    # episode):
    #
    #   combined_i = base_weight_i * obj_surrogate_i + lambda_i * obj_surrogate_i   (i < m-1)
    #   combined_{m-1} = base_weight_{m-1} * obj_surrogate_{m-1}                    (lowest priority)
    #   actor_objective = sum_i combined_i
    #
    # base_weight_i is a fixed geometric decay by priority (so a
    # higher-priority objective still gets *some* direct gradient before its
    # lambda has anything to defend - important early in training). lambda_i
    # is a persistent, dual-ascent-updated Lagrange multiplier: it grows
    # whenever objective i's estimated performance (K_i) drops more than
    # tolerance tau_i below the best level it has ratcheted up to so far,
    # and decays back toward 0 otherwise (see _update_lexicographic_duals).
    # This is the same mechanism Skalse et al. use (their eq. for L_i(theta,
    # lambda)), just evaluated once per episode instead of continuously, and
    # with a ratcheted EMA standing in for their "locked k_j after
    # convergence" step, since we have no convergence test to trigger a lock.
    def _lexicographic_actor_loss(self, obj_surrogate, mask):
        """
        obj_surrogate: (bs, T, n_agents, n_objectives) - clipped PPO surrogate
                        per objective, BEFORE combining.
        Returns (actor_objective, log_dict) where actor_objective has the
        objective dimension already collapsed: (bs, T, n_agents).
        """
        m = self.n_objectives
        combined = self.lex_base_weights[m - 1] * obj_surrogate[..., m - 1]
        log_dict = {}
        for i in range(m - 1):
            lam = self.lex_lambda[i]
            weight_i = self.lex_base_weights[i] + lam
            combined = combined + weight_i * obj_surrogate[..., i]
            log_dict[f"lex_lambda_r{i+1}"] = lam
            log_dict[f"lex_k_ref_r{i+1}"] = (self.lex_k_ema[i] if self.lex_k_ema[i] is not None else float('nan'))
        return combined, log_dict

    def _update_lexicographic_duals(self, target_returns, raw_rewards, mask):
        """
        Dual-ascent update of lex_lambda + ratcheted EMA update of lex_k_ema,
        once per episode. Skipped for any constrained objective i that was
        never "active" this episode (raw reward channel i was all zero -
        e.g. no EV ever appeared), so quiet episodes cannot look like a
        regression and erode the threshold - see the EV-sparsity discussion.
        """
        mask_sum = mask.sum().item()
        if mask_sum == 0:
            return
        for i in range(self.n_objectives - 1):
            active = (raw_rewards[..., i].abs().sum().item() > 0.0)
            if not active:
                continue
            k_now = (target_returns[..., i] * mask).sum().item() / mask_sum
            tau_i = self.lex_tolerance[i] if i < len(self.lex_tolerance) else 0.0

            if self.lex_k_ema[i] is None:
                self.lex_k_ema[i] = k_now
            else:
                ema_candidate = (1 - self.lex_ema_rho) * self.lex_k_ema[i] + self.lex_ema_rho * k_now
                # ratchet: only allow the threshold to rise (never silently
                # lower the bar), matching "remains bounded by its current
                # [best] value" in the source algorithm.
                self.lex_k_ema[i] = max(self.lex_k_ema[i], ema_candidate)

            violation = self.lex_k_ema[i] - tau_i - k_now
            self.lex_lambda[i] = max(0.0, self.lex_lambda[i] + self.lex_dual_lr * violation)
    
    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        """
        Train Critic (aligned with original project)
        NEW FOR LEXICOGRAPHIC MORL: values/returns now carry a trailing
        n_objectives dimension throughout - (bs, T, n_agents, n_objectives).
        For legacy single-objective runs n_objectives == 1 and this is
        numerically identical to the pre-LMORL code (just one extra size-1
        axis carried around).
        """
        bs = batch['batch_size']
        T = batch['max_seq_length'] - 1
        
        # 构建critic输入
        state = batch['state'][:, :-1]  # (bs, T, state_dim)
        state = state.unsqueeze(2).repeat(1, 1, self.n_agents, 1)  # (bs, T, n_agents, state_dim)
        
        agent_ids = torch.eye(self.n_agents, device=state.device).unsqueeze(0).unsqueeze(0).expand(bs, T, -1, -1)

        mask_o = mask.unsqueeze(-1)  # (bs, T, n_agents, 1) - broadcasts against the objective axis
        
        # 计算target values
        with torch.no_grad():
            state_next = batch['state'][:, 1:]  # (bs, T, state_dim)
            state_next = state_next.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            agent_ids_next = torch.eye(self.n_agents, device=state.device).unsqueeze(0).unsqueeze(0).expand(bs, T, -1, -1)
            target_vals = target_critic(state_next, agent_ids_next)  # (bs, T, n_agents, n_objectives)
        
        if self.args.get('standardise_returns', False):
            target_vals = target_vals * torch.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean
        
        # N-step returns (mask_o carries a trailing size-1 axis so this
        # broadcasts correctly against the (..., n_objectives) tensors -
        # nstep_returns() itself needs no changes, it's fully elementwise)
        target_returns = self.nstep_returns(
            rewards, mask_o, target_vals, self.args.get('q_nstep', 5)
        )
        
        if self.args.get('standardise_returns', False):
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)
        
        # 当前values
        v = critic(state, agent_ids)  # (bs, T, n_agents, n_objectives)
        
        # TD error
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask_o

        # Critic loss (averaged over agents AND objectives)
        loss = (masked_td_error ** 2).sum() / (mask.sum() * self.n_objectives)
        
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.get('max_grad_norm', 10.0)
        )
        self.critic_optimiser.step()
        
        # 统计 - per-objective, so training curves for r1/r2 (etc.) can be
        # inspected separately rather than only as a combined number.
        running_log = {
            "critic_loss": loss.item(),
            "critic_grad_norm": grad_norm.item(),
            "td_error_abs": (masked_td_error.abs().sum() / (mask.sum() * self.n_objectives)).item(),
        }
        for i in range(self.n_objectives):
            running_log[f"q_taken_mean_r{i+1}"] = (v[..., i] * mask).sum().item() / mask.sum().item()
            running_log[f"target_mean_r{i+1}"] = (target_returns[..., i] * mask).sum().item() / mask.sum().item()
        
        return masked_td_error, running_log, target_returns.detach()
    
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
                 common_reward=True,
                 # NEW FOR LEXICOGRAPHIC MORL - all optional, all default to
                 # exactly the old single-objective behaviour when left alone.
                 n_objectives=1, lexicographic=False,
                 lex_tolerance=0.0, lex_dual_lr=0.05,
                 lex_ema_rho=0.05, lex_base_weight_decay=0.1):
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_objectives = n_objectives
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
            # NEW FOR LEXICOGRAPHIC MORL
            'n_objectives': n_objectives,
            'lexicographic': lexicographic,
            'lex_tolerance': lex_tolerance,
            'lex_dual_lr': lex_dual_lr,
            'lex_ema_rho': lex_ema_rho,
            'lex_base_weight_decay': lex_base_weight_decay,
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
            rewards: (n_agents, n_objectives) numpy array (or (n_agents,) for
                     legacy single-objective callers - reshaped below so the
                     buffer always holds a consistent (n_agents, n_objectives)
                     entry regardless of which the caller passed).

        NEW FOR LEXICOGRAPHIC MORL: previously this branch built a
        `reward` variable for the common_reward case but then called
        `self.buffer.add(reward=rewards, ...)` - i.e. it ignored the
        variable it just built and stored the raw per-agent rewards
        regardless. That's removed: we always store the raw per-agent,
        per-objective reward vector. This is also *why* r1 (EV) can be a
        genuine "common"/shared reward without any common_reward special
        case here - it's already identical across agents by construction
        (see env.py World._compute_shared_lex_ev_signal), while r2 stays
        genuinely local per-agent.
        """
        obs = np.array(obs_list)  # (n_agents, obs_dim)
        state = obs.flatten()  # (obs_dim * n_agents,) 全局状态
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.ndim == 1:
            rewards = rewards[:, None]  # (n_agents,) -> (n_agents, 1)
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

        # NEW FOR LEXICOGRAPHIC MORL: persist the Lagrange multipliers and
        # ratcheted thresholds too, so resuming training doesn't forget
        # what level the EV objective had already reached.
        if getattr(self.learner, 'lexicographic', False):
            save_dict['lex_lambda'] = self.learner.lex_lambda
            save_dict['lex_k_ema'] = self.learner.lex_k_ema
        
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

        # NEW FOR LEXICOGRAPHIC MORL
        if 'lex_lambda' in checkpoint and getattr(self.learner, 'lexicographic', False):
            self.learner.lex_lambda = checkpoint['lex_lambda']
            self.learner.lex_k_ema = checkpoint['lex_k_ema']
            print(f"   └─ Lexicographic dual state restored: lambda={self.learner.lex_lambda}, "
                  f"k_ema={self.learner.lex_k_ema}")
        
        print(f"Model loaded from {path}")