import random
import torch
import torch.optim as optim
import numpy as np
from torch import nn

from data_structure import Graph, ServerNode, Microservice
from q_network import QNetwork
from replay_buffer import ReplayBuffer

# 超参数
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500


def train_dqn(graph, microservices):
    # 每个节点的内存和CPU + 边的带宽 + 目标微服务的内存使用量和吞吐量
    state_dim = len(graph.servernodes) * 2 + len(graph.edgenodes) * 2 + len(graph.edges) + 2
    # 在哪个节点部署微服务
    action_dim = len(graph.servernodes)+len(graph.edgenodes)

    # 初始化 Q 网络和目标 Q 网络
    q_network = QNetwork(state_dim, action_dim)
    target_q_network = QNetwork(state_dim, action_dim)

    # 将 Q 网络的参数复制到目标 Q 网络中
    target_q_network.load_state_dict(q_network.state_dict())

    # 初始化优化器
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    # 初始化 epsilon 和步数
    epsilon = EPSILON_START
    steps_done = 0

    # 定义选择动作的函数
    def select_action(state):
        nonlocal epsilon, steps_done
        if random.random() > epsilon:
            # 使用 Q 网络选择动作
            with torch.no_grad():
                action = q_network(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        else:
            # 随机选择动作
            action = random.randint(0, action_dim - 1)

        # 递减 epsilon
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
        steps_done += 1

        return action
    # 定义计算 TD 误差的函数
    def compute_td_loss(batch_size):
        # 从经验回放缓冲区中采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 将数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算 Q 网络的 Q 值
        q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 网络的 Q 值
        next_q_values = target_q_network(next_states).max(1)[0]

        # 计算目标 Q 值
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # 计算 MSE 损失
        loss = nn.MSELoss()(q_values, target_q_values)

        # 优化 Q 网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # 设置训练的参数
    num_episodes = 500

    # 开始训练
    for episode in range(num_episodes):
        # 重置状态,将图重置为0
        state = reset_state(graph, microservices)
        episode_reward = 0
        done = False

        while not done:
            # 选择动作
            action = select_action(state)

            # 执行动作，得到下一个状态、奖励和是否终止
            next_state, reward, done = step(graph, microservices, state, action)

            # 将经验存储到回放缓冲区中
            replay_buffer.push(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            episode_reward += reward

            # 如果回放缓冲区中的经验足够多，则进行 Q 网络的训练
            if replay_buffer.size() > BATCH_SIZE:
                loss = compute_td_loss(BATCH_SIZE)

        # 每隔 TARGET_UPDATE 轮次，将 Q 网络的参数复制到目标 Q 网络中
        if episode % TARGET_UPDATE == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        # 打印每轮的奖励
        print(f"Episode {episode}, Reward: {episode_reward}")


def reset_state(graph, microservices):
    state = []
    for node in graph.servernodes:
        state.append(node.memory)
        state.append(node.cpu)
    for node in graph.edgenodes:
        state.append(node.memory)
        state.append(node.cpu)
    for edge in graph.edges:
        state.append(edge.bandwidth)
    microservice = random.choice(microservices)
    state.append(microservice.memory_usage)
    state.append(microservice.throughput)
    return np.array(state)


def step(graph, microservices, state, action):
    # 根据state和action计算reward和next_state
    # 示例中假设next_state和reward为随机值，done为False
    next_state = state.copy()
    reward = calculate_reward(state, action)
    done = False
    return next_state, reward, done

def get_next_request():
    #这个函数中设定下一个请求出现的概率
    return
def calculate_reward(state, action):
    # 计算reward的逻辑
    return random.random()
