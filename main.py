from data_structure import Graph, ServerNode, Edge, Microservice,EdgeNode
from train import train_dqn




def initiating():
    graph = Graph()
    #初始化服务器节点
    graph.add_servernode(ServerNode(id=1,SN_id=1, memory=32, cpu=8, ms_list=[]))
    graph.add_servernode(ServerNode(id=2,SN_id=2, memory=16, cpu=16, ms_list=[]))
    graph.add_servernode(ServerNode(id=3,SN_id=3, memory=16, cpu=8, ms_list=[]))
    graph.add_servernode(ServerNode(id=4,SN_id=4, memory=32, cpu=4, ms_list=[]))
    graph.add_servernode(ServerNode(id=5,SN_id=5, memory=16, cpu=16, ms_list=[]))
    graph.add_servernode(ServerNode(id=6,SN_id=6, memory=64, cpu=12, ms_list=[]))

    #初始化服务器节点间连接情况
    graph.add_edge(1, 2, Edge(nodes=[1, 2], bandwidth=100))
    graph.add_edge(1, 3, Edge(nodes=[1, 3], bandwidth=200))
    graph.add_edge(1, 4, Edge(nodes=[1, 4], bandwidth=200))
    graph.add_edge(1, 6, Edge(nodes=[1, 6], bandwidth=300))
    graph.add_edge(2, 3, Edge(nodes=[2, 3], bandwidth=300))
    graph.add_edge(2, 5, Edge(nodes=[2, 5], bandwidth=200))
    graph.add_edge(3, 6, Edge(nodes=[3, 6], bandwidth=100))
    graph.add_edge(3, 5, Edge(nodes=[3, 5], bandwidth=100))
    graph.add_edge(4, 6, Edge(nodes=[4, 6], bandwidth=300))

    #初始化边缘节点
    graph.add_edgenode(EdgeNode(id=7,EN_id=1,SN_unique_id=1, memory=4, cpu=2,bandwith=50, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=8, EN_id=2, SN_unique_id=2, memory=8, cpu=2,bandwith=25, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=9, EN_id=3, SN_unique_id=2, memory=8, cpu=2,bandwith=100, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=10, EN_id=4, SN_unique_id=3, memory=4, cpu=2,bandwith=25, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=11, EN_id=5, SN_unique_id=3, memory=4, cpu=2,bandwith=50, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=12, EN_id=6, SN_unique_id=4, memory=8, cpu=2,bandwith=100, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=13, EN_id=7, SN_unique_id=4, memory=4, cpu=2,bandwith=100, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=14, EN_id=8, SN_unique_id=4, memory=8, cpu=2,bandwith=25, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=15, EN_id=9, SN_unique_id=6, memory=8, cpu=2,bandwith=100, ms_list=[]))
    graph.add_edgenode(EdgeNode(id=16, EN_id=10, SN_unique_id=6, memory=4, cpu=2,bandwith=50, ms_list=[]))

    #初始化微服务属性
    microservices = [
        Microservice(MS_id=1, memory_usage=4, throughput=2),
        Microservice(MS_id=2, memory_usage=8, throughput=4),
    ]
    return graph,microservices





if __name__ == "__main__":

    graph,microservices=initiating()
    graph.draw_graph()
    train_dqn(graph, microservices)
