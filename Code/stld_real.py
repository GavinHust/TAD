import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lsmr
import networkx as nx
import os
import gc
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from networkx.algorithms import centrality
from torch_geometric.data import Data
from collections import deque


# 得到营养级
def get_levels(A):
    w_in = A.sum(axis=0)  # 计算入度
    w_out = A.sum(axis=1)  # 计算出度
    u = w_in + w_out
    v = w_in - w_out
    Lambda = np.diag(u) - A - A.T
    h = lsmr(Lambda, v)[0]
    h = h - min(h)  # 保证营养级从0开始
    del Lambda, w_in, w_out, u, v
    gc.collect()
    return h


def calc_troph_incoh(A, h):
    F = 0
    if (A.sum == 0):
        return F
    idx = np.nonzero(A)
    for i in range(len(idx[0])):
        x = idx[0][i]
        y = idx[1][i]
        F = F + (h[y] - h[x] - 1) ** 2
    F = F / A.sum()
    del idx
    gc.collect()
    return F


class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.05):  # dropout=0.6
        super(ResidualGATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = lambda x: x

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.dropout(x)
        return x + res


class DeepGATNet(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features, heads_per_layer, mlp_dims):
        super(DeepGATNet, self).__init__()
        assert len(hidden_dims) == len(heads_per_layer), "Hidden dimensions and heads per layer counts must match."
        self.layers = nn.ModuleList()

        # 添加GAT层
        current_dim = in_features
        for dim, heads in zip(hidden_dims, heads_per_layer):
            self.layers.append(ResidualGATLayer(current_dim, dim, heads))
            current_dim = dim

        # 添加最后一层GATConv，不使用残差连接
        self.layers.append(GATConv(current_dim, out_features, heads=1, concat=False))

        # 添加全连接层（多层感知器）
        self.mlp = nn.Sequential(
            nn.Linear(out_features, mlp_dims[0]),
            nn.ELU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ELU(),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.Sigmoid()  # 最后一层使用Sigmoid激活函数
        )

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:  # 前面的GAT层
            x = layer(x, edge_index)
        x = self.layers[-1](x, edge_index)  # 最后一层GAT
        x = self.mlp(x)
        return x.squeeze()


def load_DND(model, optimizer, filepath):
    print(device)
    checkpoint = torch.load(filepath, map_location=device)
    # print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # 切换到评估模式
    return model, optimizer


def DND_features(G):
    features = {
        "in_degree": [degree for node, degree in G.in_degree()],
        "out_degree": [degree for node, degree in G.out_degree()],
        "betweenness": list(centrality.betweenness_centrality(G).values()),
        "pagerank": list(nx.pagerank(G).values()),
    }
    features_dict = features
    features = np.array([features_dict[key] for key in sorted(features_dict.keys())]).T

    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    edge_index = np.array(adj_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)  # 使用计算的特征
    # print(x)
    # print(edge_index)
    # print(x.shape)
    # print(edge_index.shape)
    return Data(x=x, edge_index=edge_index)


def corehd_disintegration(G):
    disintegrate_order = []
    nodes = list(G.nodes())  # 获取图的节点列表

    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)  # 原始图的邻接矩阵

    # 开始 2-core 分解
    while True:
        # 提取当前 2-core 子图
        two_core = nx.k_core(G, k=2)
        if two_core.number_of_nodes() == 0:
            break
        two_core_nodes = list(two_core.nodes())
        A_two_core = np.array(nx.adjacency_matrix(two_core).todense(), dtype=float)
        D_two_core = A_two_core.sum(axis=0) + A_two_core.sum(axis=1)
        d = np.argmax(D_two_core)
        node_to_remove = two_core_nodes[d]
        disintegrate_order.append(node_to_remove)
        G.remove_node(node_to_remove)
        for i, x in enumerate(nodes):
            if x == node_to_remove:
                indices = i
        A[:, indices] = 0
        A[indices, :] = 0

    # 处理剩余节点
    remaining_nodes = [x for x in nodes if x not in disintegrate_order]
    # print(f"Length of 2-core disintegration order: {len(disintegrate_order)}")
    # print(f"Remaining nodes: {len(remaining_nodes)}")

    # 使用广度优先搜索对剩余节点进行处理
    if remaining_nodes:
        visited = set()
        queue = deque()

        # 处理每一个剩余节点
        for node in remaining_nodes:
            node_idx = nodes.index(node)  # 查找原始节点的索引
            if node_idx not in visited:
                queue.append(node_idx)
                visited.add(node_idx)

                while queue:
                    current_node_idx = queue.popleft()
                    disintegrate_order.append(nodes[current_node_idx])

                    # 查找邻居节点并加入队列
                    for neighbor_idx in np.where(A[current_node_idx] != 0)[0]:
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)

                    # 移除当前节点的所有连接
                    A[:, current_node_idx] = 0
                    A[current_node_idx, :] = 0

    # print(f"Total disintegration order length: {len(disintegrate_order)}")
    return disintegrate_order


def scc_decomposition(graph):
    """
    使用强连通分量（SCC）分解将有向图转换为有向无环图（DAG）。
    返回 DAG 和 SCC 到节点的映射。
    """
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(graph))

    # 创建 DAG
    dag = nx.DiGraph()
    scc_to_nodes = {}

    # 为每个 SCC 创建一个超节点
    for i, scc in enumerate(sccs):
        dag.add_node(i)
        scc_to_nodes[i] = scc

    # 添加超节点之间的边
    for i, scc in enumerate(sccs):
        for node in scc:
            for neighbor in graph.neighbors(node):
                if neighbor not in scc:
                    for j, other_scc in enumerate(sccs):
                        if neighbor in other_scc:
                            dag.add_edge(i, j)
                            break

    return dag, scc_to_nodes


def control_centrality(graph):
    """
    按照层级 l_i 从高到低排序节点。
    """
    """
        移除图中的所有环，将图转换为有向无环图（DAG）。
    while True:
        try:
            # 检测并移除环
            cycle = nx.find_cycle(graph, orientation="original")
            graph.remove_edges_from(cycle)
        except nx.NetworkXNoCycle:
            # 如果没有环，退出循环
            break
    """
    # 使用 SCC 分解将图转换为 DAG
    dag, scc_to_nodes = scc_decomposition(graph)
    graph = dag

    # 计算每个节点的层级
    layer_indices = {}
    current_layer = 1
    nodes = list(graph.nodes)

    while nodes:
        # 找到当前层级的所有节点（出度为 0 的节点）
        current_layer_nodes = [node for node in nodes if graph.out_degree(node) == 0]
        # 为这些节点分配层级
        for node in current_layer_nodes:
            layer_indices[node] = current_layer
        # 移除当前层级的节点
        nodes = [node for node in nodes if node not in current_layer_nodes]
        graph.remove_nodes_from(current_layer_nodes)
        # 准备下一层级
        current_layer += 1

    # 将层级映射回原始节点
    node_layer_indices = {}
    for scc, layer in layer_indices.items():
        for node in scc_to_nodes[scc]:
            node_layer_indices[node] = layer

    # 按照层级排序
    sorted_nodes = sorted(node_layer_indices.items(), key=lambda item: item[1], reverse=True)

    # 提取排序后的节点
    sorted_nodes = [node for node, layer in sorted_nodes]

    return sorted_nodes


def findfile(directory, file_prefix):
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# dir = "new_SF/"
# dir = "DND/"
# dir = "all_graph/"


# dir = "biye_real_network/"          #真实数据
# dir = "biye_real_network_new_g/"          #真实数据
# dir = "biye_real_network_test/"          #真实数据
dir = "biye_new_g/"  # 真实数据
file_pre = "soc-Slashdot0811"  # 文件以tes_开头

filenames = findfile(dir, file_pre)
# filenames = ["SF_1000_2.8_0.3_0.63_6.34", "SF_1000_2.6_0.33_0.18", "SF_1000_3.5_0.75_0.29", "SF_1000_3.5_0.76_0.27", "SF_1000_3.6_0.91_0.9" ]
# filenames = [ "email-Eu-core"]
filenames = ["soc-Slashdot0811", "soc-Epinions1"]

epoch = 0
for file in filenames:
    print('epoch:', epoch)
    print(file)
    filename = file  # [:-4]
    g = nx.read_graphml(dir + filename)  # 读取graphhml形式储存的图
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边

    nodes = list(g.nodes)
    print(nodes)
    print(np.array(nodes))

    N = len(nodes)
    print("N", N)
    back_nodes = []
    selected = []

    # back方法选取节点                                                        #TAD方法
    print("back")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)  # 获得邻接密集矩阵
    h = get_levels(A)  # 得到营养级
    h = h.reshape((h.shape[0], 1))  # 转为列向量n*1，每一行代表对应节点的营养级
    h_h = h.T - h  # 营养级差值矩阵n*n，
    h_h = np.where(h_h >= 0, 1, h_h)  # 仅考虑back边的贡献度
    h_h = np.power((h_h - 1), 2)
    h_h_A = h_h * A  # 营养不相干程度
    back_edge = np.where(h_h_A > 1, 1, 0)
    up = h_h_A.sum(axis=1) + h_h_A.sum(axis=0)  # 所有邻居的营养级与当前节点差的和
    selected = np.argsort(up)[::-1]  # 通过up降序排序
    back_nodes = np.array(nodes)[selected]
    back_nodes = back_nodes.tolist() + list(set(g.nodes) - set(back_nodes))
    # print(len(set(back_nodes)))

    # 随机方法选取节点：                                                 #将节点顺序随机打乱
    print("rand")
    nodes_rand = list(g.nodes)
    random.shuffle(nodes_rand)

    # 适应度方法选取节点：                                                #按照节点入度与出度之和从大到小删除，每次删除一个节点就重新计算各节点的入度与出度之和
    print("HDA")
    adapt_degree = []
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)

    nodes = list(g.nodes())
    while A.sum():
        D = A.sum(axis=0) + A.sum(axis=1)
        d = np.argmax(D)
        adapt_degree.append(nodes[d])
        A[:, d] = 0  # 以前是行置零
        A[d, :] = 0
    adapt_degree = adapt_degree + list(set(g.nodes()) - set(adapt_degree))

    # 度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
    print("HD")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    D = A.sum(axis=0) + A.sum(axis=1)
    d = sorted(range(N), key=lambda k: D[k], reverse=True)
    # print(d)
    degree = np.array(list(g.nodes))[d]
    degree = degree.tolist() + list(set(g.nodes()) - set(degree))
    # print(degree)

    # MinSum方法取节点
    print("MiniSum")
    idx1 = np.load('SM_selected/' + file + '-output.npy').astype(int)
    print(idx1)
    # idx1 = np.loadtxt('SM_selected/' + file + '.npy').astype(int)
    # print(idx2)
    minisum_nodes = np.array(list(g.nodes))[idx1]
    minisum_nodes = minisum_nodes.tolist() + list(set(g.nodes()) - set(minisum_nodes))
    # print(minisum_nodes)

    # PageRank方法取节点
    # 按照PageRank的值进行ND,每次删除重新计算PageRank，还是应该计算第一次的PageRank排序
    print("PageRank")
    pagerank_nodes = []
    page_g = g.copy()
    nodes_d = len(list(page_g.nodes()))
    nodes = list(g.nodes())
    while nodes_d:
        pagerank_scores = nx.pagerank(page_g)
        # print(pagerank_scores)
        max_key = max(pagerank_scores, key=lambda k: pagerank_scores[k])
        # print(max_key)
        pagerank_nodes.append(max_key)
        page_g.remove_node(max_key)
        nodes_d = len(list(page_g.nodes()))
    pagerank_nodes = pagerank_nodes + list(set(g.nodes()) - set(pagerank_nodes))

    # DND方法取节点
    print("DND")
    mlp_dims = [100, 50, 1]  # 全连接层的神经元数量
    model_DND = DeepGATNet(in_features=4, hidden_dims=[40, 30, 20, 10], out_features=mlp_dims[0],
                           heads_per_layer=[5, 5, 5, 5], mlp_dims=mlp_dims).to(device)
    optimizer_DND = torch.optim.Adam(model_DND.parameters(), lr=0.000085)
    # model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND, 'model_checkpoint_SFandER1.pth')     #真实网络
    model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND,
                                        'model_checkpoint_SF.pth')  # model_checkpoint_SFandER在SF网络效果不行，已测试
    features_DND = DND_features(g)
    with torch.no_grad():
        model_DND.eval()
        out = model_DND(features_DND.x.to(device), features_DND.edge_index.to(device))
        # print("Test output:", out)
        # print(out.shape)
    sorted_indices_DND = torch.argsort(out, descending=True)  # 获取out节点标签预测值从大到小排列的索引值
    # print(sorted_indices_DND)
    DND_nodes = np.array(list(g.nodes))[sorted_indices_DND.tolist()]
    # DND_nodes = sorted_indices_DND.tolist()
    # print(DND_nodes)

    # FINDER方法取节点
    print("FINDER")
    idx2 = np.load('FINDER_selected_directed/' + file + '.npy').astype(int)
    print(idx2)
    nodes = [str(item) for item in idx2]
    idx2 = nodes + list(set(g.nodes()) - set(nodes))
    print(idx2)  # 有问题，所有文件的序列一样

    # CoreHD方法取点
    print("CoreHD")
    corehd_g = g.copy()
    # CoreHD_nodes = corehd_disintegration(corehd_g)
    CoreHD_nodes = corehd_disintegration(corehd_g)
    # print(CoreHD_nodes)

    # 控制中心性方法取点
    print("Control")
    control_g = g.copy()
    control_nodes = control_centrality(control_g)
    control_nodes = control_nodes + list(set(g.nodes()) - set(control_nodes))
    print(control_nodes)

    g1 = g.copy()
    g2 = g.copy()
    g3 = g.copy()
    g4 = g.copy()
    g5 = g.copy()
    g6 = g.copy()
    g7 = g.copy()
    g8 = g.copy()
    g9 = g.copy()
    g10 = g.copy()

    strong_list = []
    strong_list_rand = []
    strong_list_degree = []
    strong_list_adapt_degree = []
    strong_list_MS = []
    strong_list_pagerank = []
    strong_list_DND = []
    strong_list_FINDER = []
    strong_list_Core = []
    strong_list_control = []
    F_list = []
    F_list_rand = []
    F_list_degree = []
    F_list_adapt_degree = []
    F_list_MS = []
    F_list_pagerank = []
    F_list_DND = []
    F_list_FINDER = []
    F_list_Core = []
    F_list_control = []

    L_list = []

    file_test_list = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                      'SF_1000_3.2_0.82_0.99_5.33']
    flag = 1
    # for i in range(min(len(back_nodes),len(nodes_rand),len(degree))):
    # for i in range(len(back_nodes)):
    for i in range(len(DND_nodes)):

        strong = max(nx.strongly_connected_components(g1), key=len)
        strong_list.append(len(strong) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g1).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list.append(F)
            print("F=", F)
        # print(i, len(strong) / N)
        edges = list(g1.in_edges(back_nodes[i])) + list(g1.out_edges(back_nodes[i]))
        g1.remove_edges_from(edges)

        strong_rand = max(nx.strongly_connected_components(g2), key=len)
        strong_list_rand.append(len(strong_rand) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g2).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_rand.append(F)
            print("F=", F)
        edges = list(g2.in_edges(nodes_rand[i])) + list(g2.out_edges(nodes_rand[i]))
        g2.remove_edges_from(edges)

        strong_adapt_degree = max(nx.strongly_connected_components(g3), key=len)
        strong_list_adapt_degree.append(len(strong_adapt_degree) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g3).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_adapt_degree.append(F)
            print("F=", F)
        edges = list(g3.in_edges(adapt_degree[i])) + list(g3.out_edges(adapt_degree[i]))
        g3.remove_edges_from(edges)

        strong_degree = max(nx.strongly_connected_components(g4), key=len)
        strong_list_degree.append(len(strong_degree) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g4).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_degree.append(F)
            print("F=", F)
        edges = list(g4.in_edges(degree[i])) + list(g4.out_edges(degree[i]))
        g4.remove_edges_from(edges)

        strong_MS = max(nx.strongly_connected_components(g5), key=len)
        strong_list_MS.append(len(strong_MS) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g5).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_MS.append(F)
            print("F=", F)
        edges = list(g5.in_edges(minisum_nodes[i])) + list(g5.out_edges(minisum_nodes[i]))
        g5.remove_edges_from(edges)

        strong_pagerank = max(nx.strongly_connected_components(g6), key=len)
        strong_list_pagerank.append(len(strong_pagerank) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g6).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_pagerank.append(F)
            print("F=", F)
        edges = list(g6.in_edges(pagerank_nodes[i])) + list(g6.out_edges(pagerank_nodes[i]))
        g6.remove_edges_from(edges)

        strong_DND = max(nx.strongly_connected_components(g7), key=len)
        strong_list_DND.append(len(strong_DND) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g7).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_DND.append(F)
            print("F=", F)
        edges = list(g7.in_edges(str(DND_nodes[i]))) + list(g7.out_edges(str(DND_nodes[i])))
        g7.remove_edges_from(edges)

        strong_FINDER = max(nx.strongly_connected_components(g8), key=len)
        strong_list_FINDER.append(len(strong_FINDER) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g8).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_FINDER.append(F)
            print("F=", F)
        edges = list(g8.in_edges(str(idx2[i]))) + list(g8.out_edges(str(idx2[i])))
        g8.remove_edges_from(edges)

        strong_Core = max(nx.strongly_connected_components(g9), key=len)
        strong_list_Core.append(len(strong_Core) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g9).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_Core.append(F)
            print("F=", F)
        edges = list(g9.in_edges(CoreHD_nodes[i])) + list(g9.out_edges(CoreHD_nodes[i]))
        g9.remove_edges_from(edges)

        strong_control = max(nx.strongly_connected_components(g10), key=len)
        strong_list_control.append(len(strong_control) / N)
        if filename in file_test_list:
            A = np.array(nx.adjacency_matrix(g10).todense(), dtype=float)
            h = get_levels(A)
            F = round(calc_troph_incoh(A, h), 3)
            F_list_control.append(F)
            print("F=", F)
        edges = list(g10.in_edges(str(control_nodes[i]))) + list(g10.out_edges(str(control_nodes[i])))
        g10.remove_edges_from(edges)

        L_list.append(i / N)

        # if max(strong_list[-1], strong_list_rand[-1], strong_list_degree[-1],strong_list_adapt_degree[-1],strong_list_MS[-1]) <= 1 / N:
        # if max(strong_list[-1], strong_list_rand[-1], strong_list_degree[-1], strong_list_adapt_degree[-1],strong_list_pagerank[-1]) <= 1 / N:
        #    break

    # print(strong_list_FINDER)
    val = 1 / N

    strong_list += [val] * (N - i - 1)
    strong_list_rand += [val] * (N - i - 1)
    strong_list_degree += [val] * (N - i - 1)
    strong_list_adapt_degree += [val] * (N - i - 1)
    strong_list_MS += [val] * (N - i - 1)
    strong_list_pagerank += [val] * (N - i - 1)
    strong_list_DND += [val] * (N - i - 1)
    strong_list_FINDER += [val] * (N - i - 1)
    strong_list_Core += [val] * (N - i - 1)
    strong_list_control += [val] * (N - i - 1)

    L_list += [_ / N for _ in range(i + 1, N)]

    # y= np.linspace(0, strong_list[0], 1000)

    np.save('final_DN_result/' + filename + '_back.npy', strong_list)
    np.save('final_DN_result/' + filename + '_rand.npy', strong_list_rand)
    np.save('final_DN_result/' + filename + '_adpDegree.npy', strong_list_adapt_degree)
    np.save('final_DN_result/' + filename + '_degree.npy', strong_list_degree)
    np.save('final_DN_result/' + filename + '_MS.npy', strong_list_MS)
    np.save('final_DN_result/' + filename + '_PR.npy', strong_list_pagerank)
    np.save('final_DN_result/' + filename + '_DND.npy', strong_list_DND)
    np.save('final_DN_result/' + filename + '_finder.npy', strong_list_FINDER)
    np.save('final_DN_result/' + filename + '_Core.npy', strong_list_Core)
    np.save('final_DN_result/' + filename + '_control.npy', strong_list_control)

    # if filename in file_test_list:
    # np.save('final_F_result/' + filename + '_backF.npy', F_list)
    # np.save('final_F_result/' + filename + '_randF.npy', F_list_rand)
    # np.save('final_F_result/' + filename + '_adpDegreeF.npy', F_list_adapt_degree)
    # np.save('final_F_result/' + filename + '_degreeF.npy', F_list_degree)
    # np.save('final_F_result/' + filename + '_MSF.npy', F_list_MS)
    # np.save('final_F_result/' + filename + '_PRF.npy', F_list_pagerank)
    # np.save('final_F_result/' + filename + '_DNDF.npy', F_list_DND)
    # np.save('final_F_result/' + filename + '_FDF.npy', F_list_FINDER)
    # np.save('final_F_result/' + filename + '_CoreF.npy', F_list_Core)
    # np.save('final_F_result/' + filename + '_controlF.npy', F_list_control)

    fig, ax = plt.subplots()
    """    
    plt.plot(L_list, strong_list, color="red",lw=3)
    plt.plot(L_list, strong_list_rand,  color="blue", lw=3)
    plt.plot(L_list, strong_list_adapt_degree,  color="orange", lw=3)
    plt.plot(L_list, strong_list_degree,  color="pink", lw=3)
    plt.plot(L_list, strong_list_MS,  color="green", lw=3)
    plt.plot(L_list, strong_list_DND,  color="#00FFFF", lw=3)
    plt.plot(L_list, strong_list_pagerank,  color="yellow", lw=3)
    plt.plot(L_list, strong_list_FINDER, color="yellow", lw=3)
    plt.plot(L_list, strong_list_Core, color="yellow", lw=3)
    plt.plot(L_list, strong_list_control, color="yellow", lw=3)"""

    # plt.legend(["TAD",'HDA','HD','MinSum','Random'])
    plt.legend(['PageRank'])
    plt.xlabel("Fraction of Nodes Removed")
    plt.ylabel("Largest Strongly Connected Componenet")
    plt.title(filename)

    # plt.savefig('real_result/'+filename+'.png')
    epoch += 1
    # plt.show()
    pass

