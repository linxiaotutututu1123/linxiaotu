"""
AI预测模块 - 强化学习
使用RL进行仓位管理和策略优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from collections import deque
import random

logger = logging.getLogger(__name__)


# ==================== 强化学习配置 ====================

@dataclass
class RLConfig:
    """强化学习配置"""
    # 状态空间
    state_dim: int = 20           # 状态维度
    lookback_period: int = 30     # 回看周期
    
    # 动作空间
    action_type: str = "discrete"  # discrete / continuous
    num_actions: int = 5          # 离散动作数量 [空仓, 25%, 50%, 75%, 满仓]
    
    # 训练配置
    learning_rate: float = 0.0003
    gamma: float = 0.99           # 折扣因子
    epsilon_start: float = 1.0    # 初始探索率
    epsilon_end: float = 0.01     # 最终探索率
    epsilon_decay: float = 0.995  # 探索衰减
    
    # 经验回放
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000   # 开始训练的最小经验数
    
    # 网络配置
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # 奖励配置
    reward_type: str = "sharpe"   # return / sharpe / sortino


# ==================== 交易环境 ====================

class TradingEnvironment:
    """
    交易环境
    遵循OpenAI Gym接口
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1000000,
        commission: float = 0.0003,
        slippage: float = 0.0001,
        config: RLConfig = None
    ):
        self.config = config or RLConfig()
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        
        # 状态空间
        self.state_dim = self.config.state_dim
        self.action_space = list(range(self.config.num_actions))
        
        # 动作映射 [空仓, 25%, 50%, 75%, 满仓]
        self.action_to_position = {
            0: 0.0,
            1: 0.25,
            2: 0.5,
            3: 0.75,
            4: 1.0
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = self.config.lookback_period
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        
        # 历史记录
        self.balance_history = [self.initial_balance]
        self.position_history = [0.0]
        self.action_history = []
        self.reward_history = []
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Returns:
            state: 新状态
            reward: 奖励
            done: 是否结束
            info: 附加信息
        """
        # 获取当前价格
        current_price = self.df['close'].iloc[self.current_step]
        
        # 目标仓位
        target_position = self.action_to_position.get(action, 0.0)
        
        # 执行交易
        self._execute_trade(target_position, current_price)
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 获取新状态
        state = self._get_state() if not done else np.zeros(self.state_dim)
        
        # 记录
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step
        }
        
        return state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 价格特征
        lookback = self.config.lookback_period
        prices = self.df['close'].iloc[self.current_step-lookback:self.current_step].values
        
        # 收益率
        returns = np.diff(prices) / prices[:-1]
        state.extend([
            np.mean(returns),        # 平均收益率
            np.std(returns),         # 波动率
            returns[-1] if len(returns) > 0 else 0,  # 最新收益率
        ])
        
        # 技术指标
        # SMA
        sma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        state.append((prices[-1] - sma5) / sma5)  # 价格相对SMA5
        state.append((prices[-1] - sma20) / sma20)  # 价格相对SMA20
        
        # RSI
        delta = np.diff(prices)
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        state.append(rsi / 100)
        
        # 账户状态
        state.append(self.position)  # 当前仓位
        state.append((self.balance - self.initial_balance) / self.initial_balance)  # 账户收益率
        
        # 持仓盈亏
        if self.position > 0 and self.position_price > 0:
            pnl = (self.df['close'].iloc[self.current_step] - self.position_price) / self.position_price
        else:
            pnl = 0
        state.append(pnl)
        
        # 填充或截断到固定维度
        while len(state) < self.state_dim:
            state.append(0.0)
        state = state[:self.state_dim]
        
        return np.array(state, dtype=np.float32)
    
    def _execute_trade(self, target_position: float, price: float):
        """执行交易"""
        position_change = target_position - self.position
        
        if abs(position_change) < 0.01:
            return
        
        # 计算交易金额
        trade_value = abs(position_change) * self.balance
        
        # 考虑滑点和手续费
        if position_change > 0:  # 买入
            actual_price = price * (1 + self.slippage)
        else:  # 卖出
            actual_price = price * (1 - self.slippage)
        
        commission_cost = trade_value * self.commission
        
        # 更新持仓
        if position_change > 0:  # 买入
            self.balance -= (trade_value + commission_cost)
            self.position_price = (
                self.position_price * self.position + actual_price * position_change
            ) / (self.position + position_change) if (self.position + position_change) > 0 else actual_price
        else:  # 卖出
            # 计算盈亏
            pnl = position_change * self.balance * (price - self.position_price) / self.position_price
            self.balance += (trade_value - commission_cost + pnl)
        
        self.position = target_position
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        if len(self.balance_history) < 2:
            return 0.0
        
        # 简单收益率
        ret = (self.balance - self.balance_history[-1]) / self.balance_history[-1]
        
        if self.config.reward_type == "return":
            return ret * 100  # 放大
        
        elif self.config.reward_type == "sharpe":
            # 基于近期收益的夏普比率
            if len(self.balance_history) >= 20:
                recent_returns = np.diff(self.balance_history[-20:]) / np.array(self.balance_history[-20:-1])
                if np.std(recent_returns) > 0:
                    sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                    return sharpe
            return ret * 100
        
        elif self.config.reward_type == "sortino":
            if len(self.balance_history) >= 20:
                recent_returns = np.diff(self.balance_history[-20:]) / np.array(self.balance_history[-20:-1])
                downside = recent_returns[recent_returns < 0]
                if len(downside) > 0 and np.std(downside) > 0:
                    sortino = np.mean(recent_returns) / np.std(downside) * np.sqrt(252)
                    return sortino
            return ret * 100
        
        return ret * 100


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


# ==================== DQN Agent ====================

class DQNAgent:
    """
    Deep Q-Network Agent
    用于离散动作空间的强化学习
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        
        # 探索参数
        self.epsilon = self.config.epsilon_start
        
        # 经验回放
        self.memory = ReplayBuffer(self.config.buffer_size)
        
        # 构建网络
        self.model = None
        self.target_model = None
        self._build_networks()
    
    def _build_networks(self):
        """构建神经网络"""
        try:
            import torch
            import torch.nn as nn
            
            class DQN(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_sizes):
                    super().__init__()
                    
                    layers = []
                    prev_size = state_dim
                    for hidden_size in hidden_sizes:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.1))
                        prev_size = hidden_size
                    layers.append(nn.Linear(prev_size, action_dim))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            self.model = DQN(self.state_dim, self.action_dim, self.config.hidden_sizes)
            self.target_model = DQN(self.state_dim, self.action_dim, self.config.hidden_sizes)
            self.target_model.load_state_dict(self.model.state_dict())
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.target_model = self.target_model.to(self.device)
            
            logger.info("DQN networks built successfully")
            
        except ImportError:
            logger.warning("PyTorch not installed")
            self.model = None
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        # ε-贪婪策略
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        if self.model is None:
            return random.randint(0, self.action_dim - 1)
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()
            
            return action
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return random.randint(0, self.action_dim - 1)
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """训练一步"""
        if len(self.memory) < self.config.min_buffer_size:
            return 0.0
        
        if self.model is None:
            return 0.0
        
        try:
            import torch
            import torch.nn.functional as F
            
            # 采样
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.config.batch_size
            )
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # 当前Q值
            self.model.train()
            q_values = self.model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # 目标Q值 (Double DQN)
            with torch.no_grad():
                # 用主网络选动作
                next_actions = self.model(next_states).argmax(1)
                # 用目标网络估计Q值
                next_q_values = self.target_model(next_states)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)
            
            # 损失
            loss = F.mse_loss(q_values, target_q_values)
            
            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return 0.0
    
    def update_target_network(self):
        """更新目标网络"""
        if self.model is not None and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )


# ==================== PPO Agent ====================

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    用于连续/离散动作空间
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        
        # PPO参数
        self.clip_epsilon = 0.2
        self.c1 = 0.5  # value loss coefficient
        self.c2 = 0.01  # entropy coefficient
        
        self.model = None
        self._build_networks()
    
    def _build_networks(self):
        """构建Actor-Critic网络"""
        try:
            import torch
            import torch.nn as nn
            
            class ActorCritic(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_sizes):
                    super().__init__()
                    
                    # 共享特征提取
                    self.features = nn.Sequential(
                        nn.Linear(state_dim, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU()
                    )
                    
                    # Actor (策略网络)
                    self.actor = nn.Sequential(
                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[2], action_dim),
                        nn.Softmax(dim=-1)
                    )
                    
                    # Critic (价值网络)
                    self.critic = nn.Sequential(
                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[2], 1)
                    )
                
                def forward(self, x):
                    features = self.features(x)
                    action_probs = self.actor(features)
                    value = self.critic(features)
                    return action_probs, value
                
                def get_action(self, x):
                    action_probs, _ = self.forward(x)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    return action, log_prob
            
            self.model = ActorCritic(
                self.state_dim, 
                self.action_dim, 
                self.config.hidden_sizes
            )
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            logger.info("PPO network built successfully")
            
        except ImportError:
            logger.warning("PyTorch not installed")
            self.model = None
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """选择动作"""
        if self.model is None:
            return random.randint(0, self.action_dim - 1), 0.0
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob = self.model.get_action(state_tensor)
            
            return action.item(), log_prob.item()
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return random.randint(0, self.action_dim - 1), 0.0


# ==================== RL训练管理器 ====================

class RLTrainer:
    """
    强化学习训练管理器
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        agent_type: str = "dqn",
        config: RLConfig = None
    ):
        self.config = config or RLConfig()
        
        # 创建环境
        self.env = TradingEnvironment(df, config=self.config)
        
        # 创建Agent
        if agent_type == "dqn":
            self.agent = DQNAgent(
                self.config.state_dim,
                self.config.num_actions,
                self.config
            )
        elif agent_type == "ppo":
            self.agent = PPOAgent(
                self.config.state_dim,
                self.config.num_actions,
                self.config
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # 训练历史
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def train(
        self, 
        num_episodes: int = 1000,
        target_update_freq: int = 10,
        log_freq: int = 100
    ) -> Dict[str, List]:
        """
        训练Agent
        """
        logger.info(f"Starting RL training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                # 选择动作
                action = self.agent.select_action(state)
                if isinstance(action, tuple):
                    action = action[0]
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(state, action, reward, next_state, done)
                
                # 训练
                if hasattr(self.agent, 'train_step'):
                    self.agent.train_step()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # 更新目标网络
            if episode % target_update_freq == 0:
                if hasattr(self.agent, 'update_target_network'):
                    self.agent.update_target_network()
            
            # 衰减探索率
            if hasattr(self.agent, 'decay_epsilon'):
                self.agent.decay_epsilon()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if (episode + 1) % log_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-log_freq:])
                logger.info(f"Episode {episode+1}/{num_episodes}, "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Epsilon: {getattr(self.agent, 'epsilon', 0):.4f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估Agent"""
        total_rewards = []
        final_balances = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                if isinstance(action, tuple):
                    action = action[0]
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            final_balances.append(self.env.balance)
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_return': np.mean([
                (b - self.env.initial_balance) / self.env.initial_balance 
                for b in final_balances
            ]),
            'best_balance': max(final_balances),
            'worst_balance': min(final_balances)
        }
