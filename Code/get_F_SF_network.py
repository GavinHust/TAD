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



# 生成满足目标 F 值的有向网络
def generate_network(file, num_nodes, target_F, tolerance=0.001, max_iter=10000):
    """
    # 初始化一个稀疏的有向图
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # 随机添加初始边
    for _ in range(num_nodes * 5):  # 初始设置约为节点数两倍的有向边
        u, v = np.random.randint(0, num_nodes, size=2)
        if u != v:
            G.add_edge(u, v)
    """
    G = nx.read_graphml("SF_new_lamda_test/new/" + file)
    # 转换为邻接矩阵
    A = nx.to_numpy_array(G)
    # 计算初始营养级和 F 值
    h = get_levels(A)
    current_F = calc_troph_incoh(A, h)
    print(current_F)
    iter_count = 0

    flag = 1
    while abs(current_F - target_F) > tolerance and iter_count < max_iter:  # 等到最后会导致
        if current_F > target_F:
            candidate_edges = [(str(u), str(v)) for u, v in G.edges if h[int(u)] > h[int(v)]]
            if candidate_edges:  # 确保有符合条件的边可以删除
                u, v = candidate_edges[np.random.randint(len(candidate_edges))]
                if not G.has_edge(v, u):  # 检查反向边是否已经存在
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
        else:
            # print(h)
            candidate_edges = [(str(u), str(v)) for u, v in G.edges if h[int(u)] < h[int(v)]]
            if candidate_edges:  # 确保有符合条件的边可以删除
                u, v = candidate_edges[np.random.randint(len(candidate_edges))]
                if not G.has_edge(v, u):  # 检查反向边是否已经存在
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
            else:
                flag = 0
                print(666)

        # 更新邻接矩阵、营养级和 F 值
        A = nx.to_numpy_array(G)
        h = get_levels(A)
        current_F = calc_troph_incoh(A, h)
        iter_count += 1

        # 打印当前状态（可选）
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}: Current F = {current_F:.4f}")

    if abs(current_F - target_F) <= tolerance:
        print(f"Successfully generated directed network with F = {current_F:.4f}")
    else:
        print(f"Failed to reach target F within {max_iter} iterations. Final F = {current_F:.4f}")

    return G




# 生成满足目标 F 值的有向网络
def generate_network_show(file, num_nodes, target_F, tolerance=0.001, max_iter=10000):
    """
    # 初始化一个稀疏的有向图
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # 随机添加初始边
    for _ in range(num_nodes * 5):  # 初始设置约为节点数两倍的有向边
        u, v = np.random.randint(0, num_nodes, size=2)
        if u != v:
            G.add_edge(u, v)
    """
    #G = nx.read_graphml("SF_new_lamda_test/new_0.61/" + file)
    #G_0 = nx.read_graphml("all_graph/" + file)
    G_0 = nx.DiGraph()
    nodes = list(range(10))  # 节点编号为 1 到 10
    G_0.add_nodes_from(nodes)
    # 手动指定边
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 8),  # 从节点 1 和 2 到节点 4 的边
        (1, 8),                     # 从节点 4 到节点 5 和 6 的边
        (7, 8),(3,5),(8,6),(9,0), (0,2), (9,6)          # 从节点 7 到节点 8 的边
    ]
    # 添加边到图中
    G_0.add_edges_from(edges)
    print(G_0)
    G = G_0.copy()
    # 转换为邻接矩阵
    A = nx.to_numpy_array(G)
    # 计算初始营养级和 F 值
    h = get_levels(A)
    print(h)
    current_F = calc_troph_incoh(A, h)
    orign_F = current_F
    print(current_F)
    iter_count = 0

    flag = 1
    while abs(current_F - target_F) > tolerance and iter_count < max_iter:  # 等到最后会导致
        if current_F > target_F:
            candidate_edges = [(str(u), str(v)) for u, v in G.edges if h[int(u)] > h[int(v)]]
            if candidate_edges:  # 确保有符合条件的边可以删除
                u, v = candidate_edges[np.random.randint(len(candidate_edges))]
                if not G.has_edge(v, u):  # 检查反向边是否已经存在
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
        else:
            # print(h)
            candidate_edges = [(u, v) for u, v in G.edges if h[int(u)] < h[int(v)]]
            print(candidate_edges)
            if candidate_edges:  # 确保有符合条件的边可以删除
                u, v = candidate_edges[np.random.randint(len(candidate_edges))]
                if not G.has_edge(v, u):  # 检查反向边是否已经存在
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
            else:
                flag = 0
                print(666)

        # 更新邻接矩阵、营养级和 F 值
        A = nx.to_numpy_array(G)
        h = get_levels(A)
        current_F = calc_troph_incoh(A, h)
        iter_count += 1

        # 每次反向一条边后绘制原始网络和反向边后的网络
        # 创建一个包含两个子图的画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # 原始网络布局
        pos = nx.kamada_kawai_layout(G_0)

        # 绘制原始网络
        nx.draw(G_0, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', font_size=10,
                font_weight='bold', arrows=True, arrowsize=20, ax=ax1)
        ax1.set_title(f"Original Network\n$\\mathit{{F}}$ = {orign_F:.4f}",  fontsize=12)

        # 创建一个副本网络，反向边
        # 绘制反向边后的网络
        nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, edge_color='gray',
                font_size=10,
                font_weight='bold', arrows=True, arrowsize=20, ax=ax2)
        # 将反向边标红
        nx.draw_networkx_edges(G, pos, edgelist=[(v, u)], edge_color='red', arrows=True, arrowsize=20, width=2, ax=ax2)
        ax2.set_title(f"Reversed Edge ({u}, {v})\n$\\mathit{{F}}$ = {current_F:.4f}", fontsize=12)

        ax1.text(-0.05, 1.15, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        ax2.text(-0.05, 1.15, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        # 显示整个图
        plt.tight_layout()
        plt.show()


        # 打印当前状态（可选）
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}: Current F = {current_F:.4f}")

    if abs(current_F - target_F) <= tolerance:
        print(f"Successfully generated directed network with F = {current_F:.4f}")
    else:
        print(f"Failed to reach target F within {max_iter} iterations. Final F = {current_F:.4f}")

    return G



def generate_network_T(N, edges, TGen):  # 传入参数：节点数、边数、温度参数
    """
    论文中为随机生成一个随机网络
    G = nx.DiGraph()
    for i in range(N):
        target = random.randint(0, N - 1)
        while target == i:  # 避免自环
            target = random.randint(0, N - 1)
        G.add_edge(target, i)
    """
    G = nx.read_graphml("SF_new_lamda_test/0.55/" + "SF_1000_2.8_297_13.12_0.5471")  # 导入一个SF网络
    edges = G.number_of_edges()
    G = nx.relabel_nodes(G, lambda x: int(x))  # 将节点编号从字符型转换为整型

    # 转换为邻接矩阵并计算初始营养级
    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    trophic_levels = get_levels(A)
    # F = round(calc_troph_incoh(A, trophic_levels), 2)
    # print(F)
    # print(nx.is_directed(G))

    """
    # 论文中为移除所有边，保留节点
    for u, v in list(G.edges()):
        G.remove_edge(u, v)
    """

    # 预先生成所有可能的边及其概率
    possible_edges = [
        (i, j) for i in range(N) for j in range(N) if i != j and not G.has_edge(i, j)
    ]
    probs = np.array([np.exp(-((trophic_levels[j] - trophic_levels[i] - 1) ** 2) / (2 * TGen)) for i, j in
                      possible_edges])  # 对应边被生成的概率
    total_prob = np.sum(probs)
    # 按累积概率方式抽样边
    cumulative_probs = np.cumsum(probs)  # 累积概率
    while len(G.edges) < edges:  # 直到生成所需边数
        r = random.uniform(0, total_prob)  # 在 [0, S] 区间内生成随机数
        edge_idx = np.searchsorted(cumulative_probs, r)  # 定位对应的边索引
        edge = possible_edges[edge_idx]  # 找到对应的边
        G.add_edge(*edge)  # 添加边到网络
        probs[edge_idx] = 0  # 将添加的边概率置为 0，防止再次被添加
        total_prob = np.sum(probs)  # 重新计算累计概率
        cumulative_probs = np.cumsum(probs)  # 更新累积概率
    return G


"""
#F=0和F=1
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

#F初值在0.6之前都可以大量生成SF网络统计对应的F初值
#而F初值在0.6以上根据之前生成的幂律分布的网络，通过改变边的方向，增加网络F值
F_gen_list0 = [0.55,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.75, 0.8, 0.85, 0.9]
F_gen_list = [0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.75, 0.8, 0.85, 0.9]
F_gen_list = [0.6]
for F_gen in F_gen_list:
    i =0
    print(F_gen,F_gen_list0[F_gen_list0.index(F_gen)-1])
    filenames = findfile("SF_new_lamda_test/new_0.61/", "SF_1000_2.8_"+str(F_gen_list0[F_gen_list0.index(F_gen)-1])+"_")
    for file in filenames:
        print(F_gen, file)
        G = generate_network(file,1000, F_gen)
        print(nx.is_directed(G))
        A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
        h = get_levels(A)
        F = round(calc_troph_incoh(A, h), 2)
        node_count = G.number_of_nodes()  # 返回节点数
        edge_count = G.number_of_edges()  # 返回边数
        print("节点数:", node_count)
        print("边数:", edge_count)
        print("F:", F)
        #TGen = round(TGen, 2)
        #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen) + "_" + str(i)
        #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen)+ "_20"
        #network_name = 'SF_' + '1000_' + str(F) + '_' + str(TGen)
        network_name = 'SF_' + '1000_2.8_' + str(F) + '_' + str(i)
        i = i+1
        nx.write_graphml(G, 'SF_new_lamda_test/new_0.61/' + network_name)





"""
#绘制反向边的示意图
F_gen = 0.6
filenames = ["SF_20_2.6_4.4"]
for file in filenames:
    print(F_gen, file)
    G = generate_network_show(file,1000, F_gen)
    print(nx.is_directed(G))
    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    h = get_levels(A)
    F = round(calc_troph_incoh(A, h), 2)
    node_count = G.number_of_nodes()  # 返回节点数
    edge_count = G.number_of_edges()  # 返回边数
    print("节点数:", node_count)
    print("边数:", edge_count)
    print("F:", F)
    #TGen = round(TGen, 2)
    #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen) + "_" + str(i)
    #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen)+ "_20"
    #network_name = 'SF_' + '1000_' + str(F) + '_' + str(TGen)
    network_name = 'SF_' + '1000_2.8_' + str(F) + '_' + str(i)
    i = i+1
    #nx.write_graphml(G, 'SF_new_lamda_test/new_0.61/' + network_name)
"""


"""
#根据之前生成的幂律分布的网络，通过改变边的方向，增加网络F值
F_gen_list0 = [0.1, 0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
             0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
             0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
             0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,
             0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,
             0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,
             0.7, 0.75, 0.8, 0.85, 0.9]
F_gen_list = [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
             0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
             0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
             0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,
             0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,
             0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,
             0.7, 0.75, 0.8, 0.85, 0.9]
#F_gen_list = [0.1]
#F_gen_list0 = [0.11, 0.1]
for F_gen in F_gen_list:
    i =60
    print(F_gen,F_gen_list0[F_gen_list0.index(F_gen)-1])
    filenames = findfile("SF_new_lamda_test/new/", "SF_1000_2.8_"+str(F_gen_list0[F_gen_list0.index(F_gen)-1])+"_")
    filenames = filenames[:40]
    print(len(filenames))
    for file in filenames:
        print(F_gen, file)
        G = generate_network(file,1000, F_gen)
        print(nx.is_directed(G))
        A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
        h = get_levels(A)
        F = round(calc_troph_incoh(A, h), 2)
        node_count = G.number_of_nodes()  # 返回节点数
        edge_count = G.number_of_edges()  # 返回边数
        print("节点数:", node_count)
        print("边数:", edge_count)
        print("F:", F)
        #TGen = round(TGen, 2)
        #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen) + "_" + str(i)
        #network_name = 'F_' + '1000_' + str(F) + '_' + str(TGen)+ "_20"
        #network_name = 'SF_' + '1000_' + str(F) + '_' + str(TGen)
        network_name = 'SF_' + '1000_2.8_' + str(F) + '_' + str(i)
        i = i+1
        nx.write_graphml(G, 'SF_new_lamda_test/new_0.61/' + network_name)
"""