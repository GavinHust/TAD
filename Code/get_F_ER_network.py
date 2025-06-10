import networkx as nx
import numpy as np
from scipy.sparse.linalg import lsmr
import gc
import random
import os
import matplotlib.pyplot as plt


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
    idx = np.nonzero(A)
    for i in range(len(idx[0])):
        x = idx[0][i]
        y = idx[1][i]
        F = F + (h[y] - h[x] - 1) ** 2
    F = F / A.sum()
    del idx
    gc.collect()
    return F


def findfile(directory, file_prefix):
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


def create_f_zero_graph(n):
    G = nx.DiGraph()
    random_number = random.randint(0, 9)
    for i in range(n - 1 - random_number):
        for j in range(n - 1 - random_number, n):
            G.add_edge(i, j)
    return G


def create_f_one_matrix(n):  # 每个节点出度为1，入度为1，也可以修改，只需保证所有节点入度和出度相等即可
    G = nx.DiGraph()
    random_number = random.randint(0, 30)
    for i in range(n):
        for j in range(i + 1 + random_number, i + 1 + random_number + 10):
            G.add_edge(i, j % n)  # 基础链
    return G



def generate_network_T0(N, edges, TGen):         #计算初始营养级后删除边

    G = nx.DiGraph()
    for i in range(N):
        target = random.randint(0, N - 1)
        while target == i:  # 避免自环
            target = random.randint(0, N - 1)
        G.add_edge(target, i)


    # 转换为邻接矩阵并计算初始Trophic Levels
    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    trophic_levels = get_levels(A)
    F = round(calc_troph_incoh(A, trophic_levels), 2)
    #print(F)
    #print(nx.is_directed(G))


    # 移除所有边，保留节点
    for u, v in list(G.edges()):
        G.remove_edge(u, v)

    # 预先生成所有可能的边及其概率
    possible_edges = [
        (i, j) for i in range(N) for j in range(N) if i != j and not G.has_edge(i, j)
    ]
    probs = np.array([
        np.exp(-((trophic_levels[j] - trophic_levels[i] - 1) ** 2) / (2 * TGen))
        for i, j in possible_edges
    ])
    total_prob = np.sum(probs)
    # 按累积概率方式抽样边
    cumulative_probs = np.cumsum(probs)  # 累积概率
    while len(G.edges) < edges:
        r = random.uniform(0, total_prob)  # 在 [0, S] 区间内生成随机数
        edge_idx = np.searchsorted(cumulative_probs, r)  # 定位对应的边索引
        edge = possible_edges[edge_idx]  # 找到对应的边
        # 添加边到网络
        G.add_edge(*edge)
        # 将对应的概率置为 0，并重新计算累积概率
        probs[edge_idx] = 0
        # possible_edges.pop(edge_idx)  # 从可能边列表中移除该边
        total_prob = np.sum(probs)
        cumulative_probs = np.cumsum(probs)  # 更新累积概率
    return G



"""
#生成F=0和F=1
for n in range(30):
    # 生成F为0和1的1000个节点的网络
    #g = create_f_zero_graph(1000)
    g = create_f_one_matrix(1000)
    node_count = g.number_of_nodes()  # 返回节点数
    edge_count = g.number_of_edges()  # 返回边数
    print("节点数:", node_count)
    print("边数:", edge_count)
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    h = get_levels(A)
    F = round(calc_troph_incoh(A, h),1)
    print("F:", F)
    network_name = 'F_' + '1000_' + str(F) + '_' + str(n)
    nx.write_graphml(g, 'F_networks_new/' + network_name)
"""



# 生成不同TGen值的网络
N = 1000  # 节点数
edges = 10000  # 边数
#TGen_values = np.logspace(0.3, 2.5, 50)
TGen_values = np.linspace(0.125, 0.27, 1000)
#随机网络，删边
#TGen_values = [#0.02, 0.025, 0.03,
               #0.14, 0.15, 0.16,
               #0.29, 0.295, 0.3,
               #0.43, 0.44, 0.45,
               #0.62, 0.63, 0.64,
              #0.89, 0.91, 0.93,
               #1.29, 1.32, 1.35,
               #2.1, 2.15, 2.2,
               #6, 8, 10]         #分别对于生成的F为0.1-0.9
#随机网络，删边
#TGen_values = [0.195, 0.194, 0.1945, #0.15
               #0.37, 0.369, 0.368,       #0.25
               #0.59, 0.595, 0.592,    #0.35
               #0.935, 0.94, 0.945,
               #1.405, 1.41, 1.415,
              #2.25, 2.2, 2.15,
               #3.8, 3.9, 3.7,        #0.75
               #10, 11, 12]         #分别对于生成的F为0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85
print(len(TGen_values))
F_list = []
for TGen in TGen_values:
    #for i in range(10):
    print("TGen",TGen)
    G = generate_network_T0(N, edges, TGen)
    print(nx.is_directed(G))
    node_count = G.number_of_nodes()
    strong = max(nx.strongly_connected_components(G), key=len)
    if (len(strong) / node_count) < 0.1:
        print(len(strong) / node_count)
        continue

    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    h = get_levels(A)
    F = round(calc_troph_incoh(A, h), 2)
    F_list.append(F)
    node_count = G.number_of_nodes()  # 返回节点数
    edge_count = G.number_of_edges()  # 返回边数
    print("节点数:", node_count)
    print("边数:", edge_count)
    print("F:", F)
    #TGen = round(TGen, 2)
    #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen) + "_" + str(i)
    #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen)+ "_20"
    network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen) + "_9"
    #network_name = 'F_ER_' + '1000_' + str(F) + '_' + str(TGen) + "_" + str(i)
    #nx.write_graphml(G, 'F_ER_new_new/' + network_name)
    nx.write_graphml(G, '2_ER/' + network_name)


plt.figure()
plt.plot(TGen_values, F_list)
plt.show()
