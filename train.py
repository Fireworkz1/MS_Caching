import random

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from torch import nn

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


def train_dqn(graph, microservices, nx_graph):
    shortest_paths = compute_all_pairs_shortest_path(nx_graph)

    # 每个节点的内存和CPU和五种微服务的部署情况 + 边的带宽 + 目标微服务的内存使用量和吞吐量+请求微服务的edgenode id编号
    attribute_num = 2 + len(microservices)
    edge_num = 0
    for node1_id in graph.edges:
        for node2_id, edge in graph.edges[node1_id].items():
            edge_num += 1
    edge_num = int(edge_num / 2)
    state_dim = len(graph.servernodes) * attribute_num + len(graph.edgenodes) * attribute_num + edge_num + 4 + 1
    # 在哪个节点部署微服务
    action_dim = len(graph.servernodes) + len(graph.edgenodes)
    servernode_cursor = 0
    edgenode_cursor = len(graph.servernodes) * (2 + len(microservices))
    edge_cursor = edgenode_cursor + len(graph.edgenodes) * (2 + len(microservices))
    request_cursor = edge_cursor + edge_num
    cursor = [servernode_cursor, edgenode_cursor, edge_cursor, request_cursor]
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

        print("episode:" + str(episode + 1))
        # 重置状态,将图重置为0
        state = reset_state(graph, microservices)
        # 同时初始化图中信息
        graph.init_deployment_info(microservices)
        episode_reward = 0
        done = False
        iter = 0
        while not done:
            iter += 1
            print(iter)
            # 选择动作
            action = select_action(state)

            # 执行动作，得到下一个状态、奖励和是否终止
            next_state, reward, done = step(graph, microservices, state, action, cursor)

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
    # 6*7
    for node in graph.servernodes.values():
        state.append(node.memory)
        state.append(node.cpu)
        state.extend([0] * len(microservices))
    # 10*7
    for node in graph.edgenodes.values():
        state.append(node.memory)
        state.append(node.cpu)
        state.extend([0] * len(microservices))
    # 19*1
    for node1_id in graph.edges:
        for node2_id, edge in graph.edges[node1_id].items():
            if node1_id < node2_id:
                state.append(edge.bandwidth)
    key, microservice = random.choice(list(microservices.items()))
    # 1*4
    state.append(microservice.MS_id)
    state.append(microservice.memory_usage)
    state.append(microservice.throughput)
    state.append(microservice.calculation)
    # 1*1
    state.append(random.choice(list(graph.edgenodes.keys())))

    return np.array(state)


def step(graph, microservices, state, action, cursor):
    next_state = state.copy()
    msid = state[cursor[3] + 0]
    microservice = microservices[msid]
    if action < 6:
        # 部署在servernode
        node = graph.servernodes[action]
    else:
        # 部署在edgenode
        node = graph.edgenodes[action]
    # flag用来判断当前部署位置是否已经部署了该微服务。如果是，则不部署不替换，flag为1；如果不是则flag为0
    # flag=0.无需缓存替换，正常部署
    # flag=1,无需部署
    # flag=2,缓存替换
    # flag=3.选定的部署位置无法部署微服务（总缓存小于微服务需要缓存）

    if msid in node.ms_list:
        # 当微服务不需要部署时
        flag = 1
    else:
        # 当微服务要部署时需要进行的逻辑判断
        # 如果内存剩余空间足，无需缓存替换
        if state[action * 7 + 0] > microservice.memory_usage:
            flag = 0
            node.ms_list.append(microservice.MS_id)
            graph.edgenodes[cursor[3] + 3].coresponding_ms[msid - 1] = action
            microservice.deploy_list.append(action)
        elif node.memory < microservice.memory_usage:
            flag = 3
            # TODO:添加对应逻辑
        else:
            flag = 2
            ms_list = [0, 0, 0, 0, 0]
            ms_list[0:6] = state[action * 7 + 2:action * 7 + 8]
            new_ms_list, new_memory_usage = graph.caching_exchange_microservice(node.id, ms_list, microservices,
                                                                                microservice.MS_id)

    # 计算reward
    reward = calculate_reward(state, action)

    done = False
    # 转移到下一个状态
    # cursor=[servernode_cursor,edgenode_cursor,edge_cursor,request_cursor]
    # if next_state[action * 7 + 2 + next_state[cursor[3] + 0]] == 0:
    # 当前请求导致的状态改变
    if flag == 0:
        next_state[action * 7 + 2 + next_state[cursor[3] + 0]] = 1
        next_state[action * 7 + 0] -= microservice.memory_usage
    elif flag == 1:
        # do nothing(不部署)
    elif flag == 2:
        next_state[action * 7 + 2:action * 7 + 8] = new_ms_list[0:6]
        next_state[action * 7 + 0]=new_memory_usage
    elif flag ==3:
        # TODO:添加对应逻辑
    else:
        pass

    # 选取下一个请求并修改对于状态
    # get_next_request()
    next_microservice = random.choice(list(microservices.values()))
    next_state[cursor[3] + 0] = next_microservice.MS_id
    next_state[cursor[3] + 1] = next_microservice.memory_usage
    next_state[cursor[3] + 2] = next_microservice.calculation
    next_state[cursor[3] + 3] = random.choice(graph.edgenodes).id

    return next_state, reward, done


def get_next_request():
    # 这个函数中设定下一个请求出现的概率
    return


def calculate_reward(state, action):
    # 计算reward的逻辑
    return random.random()


def compute_all_pairs_shortest_path(nx_graph):
    n = len(nx_graph.nodes())
    map = [[{'length': None, 'path': None} for _ in range(n + 1)] for _ in range(n + 1)]
    # 使用 Floyd-Warshall 算法计算全源最短路径
    path_lengths, paths = nx.floyd_warshall_predecessor_and_distance(nx_graph)

    # 打印结果
    for source in paths:
        for target in paths[source]:
            if source != target:  # 排除自环
                path = []
                current = target
                length = 0
                while current != source:
                    if path:
                        data = nx_graph.get_edge_data(path[-1], current)
                        length += data['weight']
                    path.append(current)
                    current = path_lengths[source][current]
                data = nx_graph.get_edge_data(path[-1], current)
                length += data['weight']
                path.append(source)
                path.reverse()
                # print(f"Shortest path from {source} to {target} is {path} with length {length}")
                map[source][target] = {'length': length, 'path': path}
            else:
                map[source][target] = {'length': 0, 'path': [source]}

    return map
