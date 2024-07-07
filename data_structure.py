from collections import namedtuple, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple
#id:unique_id,用于唯一标识节点，在需要共同计算两种节点时使用。en的sn字段为其连接到哪个sn上
@dataclass
class ServerNode:
    # 节点标识号，servernode标识号，内存，cpu，微服务部署列表
    id: int
    SN_id: int
    memory: int
    cpu: int
    ms_list: List

@dataclass
class Edge:
    # 两侧节点，该路径带宽
    nodes: Tuple[int, int]
    bandwidth: int

@dataclass
class Microservice:
    # 微服务id，内存占用，吞吐量（传输需求），运算量（计算需求），部署节点编号
    MS_id: int
    memory_usage: int
    throughput: int
    calculation: int
    deploy_list: List[int]

@dataclass
class EdgeNode:
    # 节点标识号，edgenode标识号，唯一连接servernode标识号，内存，cpu，连接带宽，微服务部署列表，请求第下标i个微服务时去哪个节点
    id: int
    EN_id: int
    SN_unique_id: int
    memory: int
    cpu: int
    bandwidth: int
    ms_list: List[int]
    coresponding_ms: List[int]
class Graph:
    def  __init__(self):
        self.servernodes = defaultdict(dict)
        self.edgenodes = defaultdict(dict)
        self.edges = defaultdict(dict)
        self.microservices_deployment=defaultdict(list)
    def add_servernode(self, node):
        self.servernodes.update({node.id:node})
    def add_edge(self, node1, node2, edge):
        if node1 not in self.edges:
            self.edges[node1] = {}
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node1][node2] = edge
        self.edges[node2][node1] = edge
    def add_edgenode(self,node):
        self.edgenodes.update({node.id:node})
        self.add_edge(node.id, node.SN_unique_id,Edge(nodes=(node.id, node.SN_unique_id), bandwidth=node.bandwidth))


    def deploy_microservice(self, node_id, microservice):
        self.microservices_deployment[node_id].append(microservice)
        print("deploy MS {} in Node {}",microservice.MS_id,node_id)

    def caching_exchange_microservice(self, node_id, microservice):
        return
    def init_deployment_info(self,microservices):
        for node in self.edgenodes.values():
            node.ms_list = []
            node.coresponding_ms = [0,0,0,0,0]
        for node in self.servernodes.values():
            node.ms_list = []
        for ms in microservices.values():
            ms.deploy_list=[]
    def draw_graph(self):
        # 创建NetworkX图
        nx_graph = nx.Graph()
        for node in self.servernodes.values():
            nx_graph.add_node(node.id, type="servernode", memory=node.memory, cpu=node.cpu)
        for node in self.edgenodes.values():
            nx_graph.add_node(node.id, type="edgenode", memory=node.memory, cpu=node.cpu)
        for node1 in self.edges:
            for node2 in self.edges[node1]:
                bandwidth = self.edges[node1][node2].bandwidth
                nx_graph.add_edge(node1, node2, bandwidth=bandwidth)

        # 创建标签
        node_labels = {}
        for node in self.servernodes.values():
            microservices = self.microservices_deployment[node.id]
            ms_info = '\n'.join(
                [f'MS{ms.MS_id}(M:{ms.memory_usage}, T:{ms.throughput})' for ms in enumerate(microservices)])
            if ms_info is None:
                ms_info = []
            node_labels[node.id] = f'SN {node.SN_id}\nMemory: {node.memory}\nCPU: {node.cpu}\n{ms_info}'

            for node in self.edgenodes.values():
                microservices = self.microservices_deployment[node.id]
                ms_info = '\n'.join(
                    [f'MS{ms.MS_id}(M:{ms.memory_usage}, T:{ms.throughput})' for ms in enumerate(microservices)])
                if ms_info is None:
                    ms_info = []
                node_labels[node.id] = f'EN {node.EN_id}\nMemory: {node.memory}\nCPU: {node.cpu}\n{ms_info}'
        # 绘制图
        fixed_positions = {
            # servernode
            1: (0, 0),
            2: (1, 1.4),
            3: (-1, 1.7),
            4: (0.8, -1),
            5: (0.3, 2.5),
            6: (-0.7, -1.8),
            # edgenode
            7: (0.7, 0.1),  # 1
            8: (1.6, 2.3),  # 2
            9: (1.9, 1.2),  # 2
            10: (-1.5, 2.5),  # 3
            11: (-1.6, 1.4),  # 3
            12: (1.4, -0.4),  # 4
            13: (1.8, -1),  # 4
            14: (1.4, -1.6),  # 4
            15: (-0.2, -2.6),  # 6
            16: (-1.2, -2.5)  # 6
        }
        node_colors = {
            'servernode': 'lightblue',
            'edgenode': 'lightgreen'
        }
        node_size = {
            'servernode': 3000,
            'edgenode': 1000
        }
        font_size = {
            'servernode': 8,
            'edgenode': 5
        }
        colors = [node_colors[nx_graph.nodes[n]['type']] for n in nx_graph.nodes()]
        sizes = [node_size[nx_graph.nodes[n]['type']] for n in nx_graph.nodes()]
        font = [font_size[nx_graph.nodes[n]['type']] for n in nx_graph.nodes()]
        labels = nx.get_edge_attributes(nx_graph, 'bandwidth')
        nx.draw(nx_graph, fixed_positions, with_labels=True, labels=node_labels, node_color=colors, node_size=sizes,
                font_size=8, font_weight='bold')
        nx.draw_networkx_edge_labels(nx_graph, fixed_positions, edge_labels=labels)
        plt.show()