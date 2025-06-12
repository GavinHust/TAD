import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.sparse.linalg import lsmr
import networkx as nx
import gc
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from networkx.algorithms import centrality
from torch_geometric.data import Data
from collections import deque
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 得到营养级
def get_levels(A):
    w_in = A.sum(axis=0)  # 计算入度
    w_out = A.sum(axis=1)  # 计算出度
    # w_in = A.sum(axis=0).A1 # 计算入度
    # w_out = A.sum(axis=1).A1  # 计算出度
    u = w_in + w_out
    v = w_in - w_out
    Lambda = np.diag(u) - A - A.T
    # Lambda = (diags(u) - A - A.T).tocsc()
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
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames

def draw_curve():   #绘制单个网络的瓦解曲线
    """
    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'Metabolic-SC.s','Metabolic_net_TH','FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter','Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food']   #fig3中的真实曲线

    network_names = ['SF_500_3.6_0.74_0.25','SF_1000_2.6_0.33_0.18','SF_1000_3.6_0.91_0.9','SF_500_3.3_0.29_0.62']  #fig5中的瓦解曲线

    network_name = network_names[0]   #选择绘制哪一个网络
    """

    network_names = [
        'FoodWebs_little_rock', "FoodWebs_Weddel_sea", "subelj_cora.e", 'p2p-Gnutella08',
        "Wiki-Vote", 'p2p-Gnutella06', "ia-crime-moreno", "FoodWebs_reef", "out.moreno_blogs_blogs",
        'Neural_net_celegans_neural', "net_green_eggs",
        'Social-leader2Inter',
        # "out.maayan-faa",
        'Neural_rhesus_brain_1',
        "Trade_net_trade_basic", 'Trade_net_trade_food',
    ]

    lamb = ['Food Webs01', 'Food Webs02', "Scholarly01",
            'p2p08', "Wiki-Vote", 'p2p06', "Crime", "Food Webs03", 'PolBlogs', 'Neural01', "Language",
            "Social",
            'Neural02', 'Trade01', 'Trade02',
            'Average']
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    fig, axes = plt.subplots(5, 3, figsize=(12, 10))
    # 自动调整布局
    #plt.tight_layout()
    fig.subplots_adjust(top=0.96, bottom=0.12, left=0.06, right=0.99, hspace=0.4, wspace=0.25)
    #network_names=['TexasPowerGrid','SF_500_3.6_0.74_0.25','SF_1000_2.6_0.33_0.18','SF_1000_3.2_0.44_0.13','SF_1000_3.6_0.91_0.9','SF_500_3.3_0.29_0.62','Neural_rhesus_brain_2','FoodWebs_reef','SF_500_3.4_0.6_0.61','Trade_net_trade_basic']
    #name = network_names[0]

    #network_name = 'p2p-Gnutella06'
    #name = "p2p-Gnutella06"
    #network_name = 'FoodWebs_Lough_Hyne'
    #name = "FoodWebs_Lough_Hyne"
    for i in range(len(network_names)):
        network_name = network_names[i]
        name = lamb[i]

        back = np.load('../Data/DNdata/NPY/Real/' + network_name + '_back.npy')
        #back = back / back[0]
        degree = np.load('../Data/DNdata/NPY/Real/' + network_name + '_degree.npy')
        #degree = degree / degree[0]
        adpdegree = np.load('../Data/DNdata/NPY/Real/' + network_name + '_adpDegree.npy')
        #adpdegree = adpdegree / adpdegree[0]
        finder = np.load('../Data/DNdata/NPY/Real/' + network_name + '_finder.npy')
        #finder = finder / finder[0]
        learn = np.load('../Data/DNdata/NPY/Real/' + network_name + '_MS.npy')
        #learn = learn / learn[0]
        prank = np.load('../Data/DNdata/NPY/Real/' + network_name + '_PR.npy')
        #prank = prank / prank[0]
        DND = np.load('../Data/DNdata/NPY/Real/' + network_name + '_DND.npy')
        #DND = DND / DND[0]
        CoreHD = np.load('../Data/DNdata/NPY/Real/' + network_name + '_Core.npy')
        #CoreHD = CoreHD / CoreHD[0]

        print(name,back.sum()/len(back))
        print(1-back.sum()/learn.sum(),1-back.sum()/finder.sum(),1-back.sum()/prank.sum(),1-back.sum()/adpdegree.sum(),1-back.sum()/degree.sum())

        x = [_ / len(back) for _ in range(len(back))]
        show_n=int(len(back)*0.6)
        # col.set_title(r'AvgD='+D[epoch],y=0.9)
        #plt.title(network_name, y=1.01, x=0.5,size=13)

        # 计算行和列索引
        row_index = i // 3  # 每行3个子图
        col_index = i % 3  # 每行的列索引
        axes[row_index, col_index].set_title(name, y=1.0, x=0.5,size=14)
        axes[row_index, col_index].plot(x[:show_n], back[:show_n], color='#403990', lw=1.8)
        axes[row_index, col_index].plot(x[:show_n], CoreHD[:show_n], color="#888888", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], prank[:show_n], color="#00FF00", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], learn[:show_n], color="#80A6E2", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], finder[:show_n], color="#FBDD85", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], DND[:show_n], color="#00FFFF", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], adpdegree[:show_n], color="#F46F43", lw=1.2)
        axes[row_index, col_index].plot(x[:show_n], degree[:show_n], color="#CF3D3E", lw=1.2)
        axes[row_index, col_index].tick_params(labelsize=10)
        #axes[row_index, col_index].set_xticks(fontsize=14)
        #axes[row_index, col_index].set_yticks(fontsize=14)
        # fig.text(0.55, 0.02, 'Fraction of Nodes Removed', fontsize=16, ha='center')
        # 添加文本标签
        if row_index == 4:
            axes[row_index, col_index].text(0.48, -0.42, 'Fraction of Nodes Removed', fontsize=12, ha='center',
                                            transform=axes[row_index, col_index].transAxes)
        axes[row_index, col_index].text(-0.18, 0.55, 'GSCC', va='center', fontsize=12, rotation='vertical',
                                        transform=axes[row_index, col_index].transAxes)
        #plt.savefig('final_result/' + network_name + '.svg')
    #fig.subplots_adjust(hspace=0.4, wspace=0.25)
    # plt.legend(["STLD", 'MinSum', 'FINDER', 'PageRank', 'HDA', "HD", "DND", "CoreHD"], prop={ 'size': 10})
    plt.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 11},
               bbox_to_anchor=(0.5, -0.55), loc=1, ncol=9, borderaxespad=0)
    plt.show()


def draw_heatmap():
    network_names = [
        'FoodWebs_little_rock', "FoodWebs_Weddel_sea", "subelj_cora.e", 'p2p-Gnutella08',
        "Wiki-Vote", 'p2p-Gnutella06', "ia-crime-moreno", "FoodWebs_reef", "out.moreno_blogs_blogs",
        'Neural_net_celegans_neural', "net_green_eggs",
        'Social-leader2Inter',
        # "out.maayan-faa",
        'Neural_rhesus_brain_1',
        "Trade_net_trade_basic", 'Trade_net_trade_food',
        'Average'
    ]
    methods = ['TAD', 'CoreHD', 'PageRk', 'MinSum', 'FINDER', 'DND', 'HDA', 'HD']
    lamb = ['Food\nWebs01', 'Food\nWebs02', "Scholarly01",
            'p2p08', "Wiki-Vote", 'p2p06', "Crime", "Food\nWebs03", 'PolBlogs', 'Neural01', "Language",
            "Social",
            'Neural02', 'Trade01', 'Trade02',
            'Average']
    # 数据矩阵
    data = np.zeros(shape=(len(methods), len(network_names)))

    # 填充数据矩阵
    for epoch in range(len(network_names) - 1):
        network_name = network_names[epoch]
        print('网络名称：', network_name)
        back = np.load('../Data/DNdata/NPY/Real/' + network_name + '_back.npy')
        # back = back / back[0]
        degree = np.load('../Data/DNdata/NPY/Real/' + network_name + '_degree.npy')
        # degree = degree / degree[0]
        adpdegree = np.load('../Data/DNdata/NPY/Real/' + network_name + '_adpDegree.npy')
        # adpdegree = adpdegree / adpdegree[0]
        finder = np.load('../Data/DNdata/NPY/Real/' + network_name + '_finder.npy')
        # finder = finder / finder[0]
        learn = np.load('../Data/DNdata/NPY/Real/' + network_name + '_MS.npy')
        # learn = learn / learn[0]
        prank = np.load('../Data/DNdata/NPY/Real/' + network_name + '_PR.npy')
        # prank = prank / prank[0]
        DND = np.load('../Data/DNdata/NPY/Real/' + network_name + '_DND.npy')
        # DND = DND / DND[0]
        CoreHD = np.load('../Data/DNdata/NPY/Real/' + network_name + '_Core.npy')
        # CoreHD = CoreHD / CoreHD[0]
        print(len(back))
        print(back[0])
        """
        data[0, epoch] = round(back.sum() / len(back), 4)
        data[1, epoch] = round(CoreHD.sum() / len(back), 4)
        data[2, epoch] = round(prank.sum() / len(back), 4)
        data[3, epoch] = round(learn.sum() / len(back), 4)
        data[4, epoch] = round(finder.sum() / len(back), 4)
        data[5, epoch] = round(DND.sum() / len(back), 4)
        data[6, epoch] = round(adpdegree.sum() / len(back), 4)
        data[7, epoch] = round(degree.sum() / len(back), 4)
        """

        """back[back < 0.1] = 0
        degree[degree < 0.1] = 0
        finder[finder < 0.1] = 0
        learn[learn < 0.1] = 0
        adpdegree[adpdegree < 0.1] = 0
        prank[prank < 0.1] = 0
        DND[DND < 0.1] = 0
        CoreHD[CoreHD < 0.1] = 0"""

        data[0, epoch] = back.sum() / len(back)
        data[1, epoch] = CoreHD.sum() / len(back)
        data[2, epoch] = prank.sum() / len(back)
        data[3, epoch] = learn.sum() / len(back)
        data[4, epoch] = finder.sum() / len(back)
        data[5, epoch] = DND.sum() / len(back)
        data[6, epoch] = adpdegree.sum() / len(back)
        data[7, epoch] = degree.sum() / len(back)

    # 添加平均值行
    data[:, len(network_names) - 1] = data.mean(axis=1)

    # 复制原始数据，用于热力图的标注
    original_data_for_annot = data.copy()

    # 创建一个与 data 相同形状的归一化数据矩阵，用于颜色映射
    normalized_data_for_heatmap_color = np.zeros_like(data)

    # 对每一列进行归一化
    for col_idx in range(data.shape[1]):
        col_data = data[:, col_idx]

        # 排除NaN值，只对有效数据进行归一化
        valid_col_data = col_data[~np.isnan(col_data)]

        if valid_col_data.size > 0:
            min_val = np.min(valid_col_data)
            max_val = np.max(valid_col_data)

            if max_val == min_val:
                # 如果所有值都相同，设置为中性颜色 (0.5)
                normalized_data_for_heatmap_color[:, col_idx] = 0.5
            else:
                # 归一化到 [0, 1] 区间
                normalized_data_for_heatmap_color[:, col_idx] = (col_data - min_val) / (max_val - min_val)
        else:
            # 如果整列都是NaN或空，设置为NaN，这样在热力图中不会着色
            normalized_data_for_heatmap_color[:, col_idx] = np.nan

    # 绘制热力图
    plt.figure(figsize=(12, 7))
    # ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues_r", cbar=False, xticklabels=lamb, yticklabels=methods,  annot_kws={"color": "black"})
    ax = sns.heatmap(normalized_data_for_heatmap_color, annot=original_data_for_annot, fmt=".3f", cmap="Blues_r",
                     cbar=False, xticklabels=lamb, yticklabels=methods,
                     linewidths=0, annot_kws={"color": "black", "fontsize": 12}, vmin=0, vmax=1.5, alpha=1)

    """# 添加列边框
    for i in range(len(network_names)+1):
        ax.axvline(x=i, color='black', linewidth=1)
    for i in range(data.shape[1]):  # 遍历每一列
        min_value = np.min(data[:, i])  # 找到当前列的最小值
        for j in range(data.shape[0]):  # 遍历当前列的每一行
            if data[j, i] == min_value:
                # 绘制红色矩形覆盖最小值区域
                rect = Rectangle((i, j), 1, 1, facecolor='#ff9999', edgecolor='none', zorder=1, alpha=1)
                ax.add_patch(rect)"""

    # plt.xlabel("Networks")
    # plt.ylabel("Methods")
    plt.title("AUC values of different dismantling methods in real networks", fontsize=14)
    plt.xticks(rotation=60, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()


# 绘制拓扑图并标出 剩余的节点和边
def draw_topology_with_lsc_STLD(g, lsc_nodes, node_remove, node_remain, ax, method_name, i0, nodes_to_remove, gscc0 , label=None, back_nodes=None):
    g_copy = g.copy()
    # 计算每个节点的度
    degrees = dict(g.degree())
    #degrees = dict(g.out_degree())
    # 根据节点的度对节点进行排序，度大的节点排在前面
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

    print(len(lsc_nodes))
    print("lsc_nodes", lsc_nodes)
    print(node_remove)
    #pos = nx.random_layout(g, seed=40)

    filename = str(i0) + "_TAD.pkl"
    #filename = str(i0) + ".pkl"
    if os.path.exists(filename):
        # 加载节点位置
        with open(filename, 'rb') as f:
            pos, g = pickle.load(f)
    else:
        # 初始化节点位置
        pos = {}
        # 计算圆的半径范围
        max_radius = 0.42  # 最大半径
        min_radius = 0.01  # 最小半径

        sorted_nodes = back_nodes  #按照STLD值排序
        # 根据back_nodes的顺序分配节点位置
        for i, node in enumerate(sorted_nodes):
            # 计算半径，列表中越靠前的节点半径越小，越靠后的节点半径越大
            if(i<len(sorted_nodes)/2):
                radius = min_radius + (max_radius - min_radius) * (i / (len(sorted_nodes) - 1 if len(sorted_nodes) > 1 else 1))*0.8
            else:
                radius = min_radius + (max_radius - min_radius) * (i / (len(sorted_nodes) - 1 if len(sorted_nodes) > 1 else 1))*0.5
            # 计算半径，列表中越靠前的节点半径越大，越靠后的节点半径越小
            #radius = min_radius + (max_radius - min_radius) * (1 - i / len(sorted_nodes))
            # 将节点放置在圆上，均匀分布
            angle = 2 * np.pi * np.random.rand()  # 随机角度以避免重叠
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        with open(filename, 'wb') as f:
            pickle.dump((pos, g), f)

    #node_colors = ['lightcoral' if node in lsc_nodes else 'gray' for node in g.nodes()]
    node_colors = []
    nodes_to_remove_set = set()
    for node in g.nodes():
        if node == node_remove:
            node_colors.append('#ff0000')
        elif node in lsc_nodes:
            node_colors.append('#ff8888')   #lightcoral#ff6666      "#b3d9ff", "#0077b6"
        elif node in node_remain:
            node_colors.append('#b3d9ff')
        else:
            #node_colors.append('gray')
            nodes_to_remove_set.add(node)
    # 移除灰色节点
    if nodes_to_remove_set:
        g.remove_nodes_from(nodes_to_remove_set)

    # 加载背景图片
    #img_path = "foodweb.png"
    if i0 ==0:
        img_path = "celegans_neural.png"
        # 调整 x 坐标范围到 (0, 0.5)
        pos = {node: (0.5 + pos[node][0], 0.5 + pos[node][1]) for node in pos}
        node_sizes = [2 if node in lsc_nodes else 1 for node in g.nodes()]
        # 如果提供了back_nodes，则根据back_nodes调整节点大小
        if back_nodes is not None:
            max_size = 50  # 列表中节点的最大大小
            default_size = 1  # 列表外节点的默认大小
            node_sizes = []
            for node in g.nodes():
                if node in back_nodes:
                    # 根据节点在back_nodes列表中的索引来确定大小
                    size = max_size - (back_nodes.index(node) * (max_size / len(back_nodes))) + default_size
                    #size = max_size * (0.9 ** back_nodes.index(node)) + default_size
                    if(node == node_remove):
                        size = size +10
                    node_sizes.append(size)
                else:
                    node_sizes.append(default_size)

    else:
        img_path = "foodweb.png"
        # 调整 x 坐标范围到 (0, 0.5)
        pos = {node: (0.5 + pos[node][0], 0.5 + pos[node][1]) for node in pos}
        node_sizes = [20 if node in lsc_nodes else 15 for node in g.nodes()]

        # 如果提供了back_nodes，则根据back_nodes调整节点大小
        if back_nodes is not None:
            max_size = 500  # 列表中节点的最大大小
            default_size = 10  # 列表外节点的默认大小
            node_sizes = []
            for node in g.nodes():
                if node in back_nodes:
                    # 根据节点在back_nodes列表中的索引来确定大小
                    #size = max_size - (back_nodes.index(node) * (max_size / len(back_nodes))) + default_size
                    size = max_size * (0.9 ** back_nodes.index(node)) + default_size
                    if(node == node_remove):
                        size = size +50
                    node_sizes.append(size)
                else:
                    node_sizes.append(default_size)

    #img = mpimg.imread(img_path)
    # 设置图片的显示范围（根据你的拓扑图坐标范围调整）
    #img_extent = (-0.05, 1.05, -0.05, 1.05)  # (x_min, x_max, y_min, y_max)
    # 在当前子图中显示图片
    #ax.imshow(img, extent=img_extent, aspect="auto", alpha=1)  # alpha 控制透明度
    print(node_sizes)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.8)
    # 绘制边
    edges = [(edge[0], edge[1]) for edge in g.edges() if edge[0] in lsc_nodes and edge[1] in lsc_nodes]
    nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color='lightcoral', alpha=0.5, ax=ax)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(f"GSCC_Size:{100*len(lsc_nodes)/len(g_copy.nodes()):.1f}%\nRemoved Nodes:{nodes_to_remove*100:.0f}%", fontsize=12)
    #ax.set_title(f"GSCC_Size:{100 * len(lsc_nodes) / gscc0:.1f}%\nRemoved Nodes:{nodes_to_remove * 100:.0f}%", fontsize=10)
    #ax.text(0.5, -0.1, f"Removed Nodes:{nodes_to_remove * 100:.0f}%\nGSCC_Size:{len(lsc_nodes)}",fontsize=10, ha='center', va='top', transform=ax.transAxes)
    if label:
        #ax.text(-0.1, 1.25, label, fontsize=12, ha='left', fontweight='bold', va='top', transform=ax.transAxes)
        ax.text(-0.02, 1.1, label, fontsize=12, ha='left', fontweight='bold', va='top', transform=ax.transAxes)


def draw_curve_new_STLD():   #绘制单个网络的瓦解曲线
    """
    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'Metabolic-SC.s','Metabolic_net_TH','FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter','Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food']   #fig3中的真实曲线
    network_name = network_names[0]   #选择绘制哪一个网络
    """

    dir = "../Data/Real/"
    #fig, axes = plt.subplots(2, 5, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})
    #fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(4, 6, figure=fig, width_ratios=[1, 0.5, 0.1, 1, 1, 1], height_ratios=[1, 1, 1,  1])
    # 合并第一列为一个大子图
    ax1 = fig.add_subplot(gs[:2, 3])
    ax2 = fig.add_subplot(gs[:2, 4])
    ax3 = fig.add_subplot(gs[:2, 5])
    ax4 = fig.add_subplot(gs[:2, :2])

    ax5 = fig.add_subplot(gs[2:, 3])
    ax6 = fig.add_subplot(gs[2:, 4])
    ax7 = fig.add_subplot(gs[2:, 5])
    ax8 = fig.add_subplot(gs[2:, :2])

    axes = [[ax1, ax2, ax3], [ax5, ax6, ax7]]
    axesc = [ax4, ax8]

    network_name_list = ['Neural_net_celegans_neural', 'FoodWebs_reef']
    name_list = ["NeuralNet_Celegans", "FoodWebs_reef"]

    """
    back = np.load('final_DN_result/' + network_name + '_back.npy')
    back = back / back[0]
    degree = np.load('final_DN_result/' + network_name + '_degree.npy')
    degree = degree / degree[0]
    adpdegree = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
    adpdegree = adpdegree / adpdegree[0]
    finder = np.load('final_DN_result/' + network_name + '_finder.npy')
    finder = finder / finder[0]
    learn = np.load('final_DN_result/' + network_name + '_MS.npy')
    learn = learn / learn[0]
    prank = np.load('final_DN_result/' + network_name + '_PR.npy')
    prank = prank / prank[0]
    # rand = np.load('final_DN_result/' + network_name + '_rand.npy')
    DND = np.load('final_DN_result/' + network_name + '_DND.npy')
    DND = DND / DND[0]
    CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
    CoreHD = CoreHD / CoreHD[0]
    # control = np.load('final_DN_result/' + network_name + '_control.npy')
    # control = control / control[0]
    """
    for i0, network_name in enumerate(network_name_list):
        name = name_list[i0]
        print(name)

        # 绘制拓扑图
        g = nx.read_graphml(dir + network_name)
        g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
        nodes = list(g.nodes)
        N = len(nodes)
        print(N)
        lscc0 = len(max(nx.strongly_connected_components(g), key=len))

        edges = g.edges()
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
        print(A)
        print(B)
        if network_name == "Neural_net_celegans_neural":
            nodes_to_remove_list = [int(N * 0.05),  int(N * 0.14), int(N * 0.18)]
            nodes_to_remove_list1 = [0.05, 0.14, 0.18]
        elif network_name == 'FoodWebs_reef':
            nodes_to_remove_list = [int(N * 0.05), int(N * 0.12), int(N * 0.18)]
            nodes_to_remove_list1 = [0.05, 0.12, 0.18]

        sample_points = []

        # back方法选取节点                                                        #STLD方法
        print("back")
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)         #获得邻接密集矩阵
        h = get_levels(A)           # 得到营养级
        h=h.reshape((h.shape[0],1))         #转为列向量n*1，每一行代表对应节点的营养级
        h_h=h.T-h                           #营养级差值矩阵n*n，
        h_h=np.where(h_h>=0,1,h_h)  #仅考虑back边的贡献度
        h_h=np.power((h_h-1),2)
        h_h_A=h_h*A                 #营养不相干程度
        back_edge=np.where(h_h_A>1,1,0)
        up=h_h_A.sum(axis=1)+h_h_A.sum(axis=0)          #所有邻居的营养级与当前节点差的和
        selected=np.argsort(up)[::-1]           #通过up降序排序
        back_nodes=np.array(nodes)[selected]
        print(len(back_nodes))
        print(back_nodes)
        back_nodes=back_nodes.tolist()+list(set(g.nodes)-set(back_nodes))
        for j,nodes_to_remove in enumerate(nodes_to_remove_list):
            back_nodes_remove = back_nodes[:nodes_to_remove]
            #print(len(set(back_nodes)))
            g_copy = g.copy()
            g_copy.remove_nodes_from(back_nodes_remove)
            lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
            print(lsc_nodes)
            fraction_removed = nodes_to_remove / len(g.nodes())
            gsc_size = len(lsc_nodes) / len(g.nodes())
            #gsc_size = len(lsc_nodes) / lscc0
            sample_points.append({"fraction_removed": fraction_removed, "gsc_size": gsc_size})
            label = chr(98 + i0*4 + j)
            g_copy=g.copy()
            #draw_topology_with_lsc_STLD_new(g, lsc_nodes,  back_nodes[nodes_to_remove],back_nodes[nodes_to_remove:], axes[i0][j], "STLD", i0, nodes_to_remove_list1[j], lscc0, label, back_nodes)
            draw_topology_with_lsc_STLD(g_copy, lsc_nodes, back_nodes[nodes_to_remove], back_nodes[nodes_to_remove:], axes[i0][j], "STLD", i0, nodes_to_remove_list1[j], lscc0, label, back_nodes)

        # 适应度方法选取节点：                                                #按照节点入度与出度之和从大到小删除，每次删除一个节点就重新计算各节点的入度与出度之和
        print("HDA")
        adapt_degree=[]
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        nodes=list(g.nodes())
        while A.sum():
            D = A.sum(axis=0) + A.sum(axis=1)
            d=np.argmax(D)
            adapt_degree.append(nodes[d])
            A[:, d] = 0  # 以前是行置零
            A[d,:] = 0
        print(len(adapt_degree))
        adapt_degree = adapt_degree + list(set(g.nodes()) - set(adapt_degree))


        # 度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
        print("HD")
        A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
        D=A.sum(axis=0)+A.sum(axis=1)
        d = sorted(range(N), key=lambda k: D[k], reverse=True)
        degree=np.array(list(g.nodes))[d]
        print(len(degree))
        degree=degree.tolist()+list(set(g.nodes())-set(degree))


        # MinSum方法取节点
        print("MiniSum")
        idx1 = np.load('../Data/Otherdata/SM_selected/' + network_name + '-output.npy').astype(int)
        print(idx1)
        #idx2 = np.loadtxt('SM_selected/' + file + '.npy').astype(int)
        #print(idx2)
        minisum_nodes = np.array(list(g.nodes))[idx1]
        print(minisum_nodes)
        print(len(minisum_nodes))
        minisum_nodes = minisum_nodes.tolist() + list(set(g.nodes()) - set(minisum_nodes))
        print(minisum_nodes)

        # PageRank方法取节点
        # 按照PageRank的值进行ND,每次删除重新计算PageRank，还是应该计算第一次的PageRank排序
        print("PageRank")
        pagerank_nodes=[]
        page_g = g.copy()
        nodes_d = len(list(page_g.nodes()))
        nodes = list(g.nodes())
        while nodes_d:
            pagerank_scores = nx.pagerank(page_g)
            max_key = max(pagerank_scores, key=lambda k: pagerank_scores[k])
            pagerank_nodes.append(max_key)
            page_g.remove_node(max_key)
            nodes_d = len(list(page_g.nodes()))
        print(len(pagerank_nodes))
        pagerank_nodes = pagerank_nodes + list(set(g.nodes()) - set(pagerank_nodes))


        # DND方法取节点
        print("DND")
        mlp_dims = [100, 50, 1]  # 全连接层的神经元数量
        model_DND = DeepGATNet(in_features=4, hidden_dims=[40, 30, 20, 10], out_features=mlp_dims[0],
                               heads_per_layer=[5, 5, 5, 5], mlp_dims=mlp_dims).to(device)
        optimizer_DND = torch.optim.Adam(model_DND.parameters(), lr=0.000085)
        # model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND, 'model_checkpoint_SFandER1.pth')     #真实网络
        model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND,
                                            'model_checkpoint_DND.pth')  # model_checkpoint_SFandER在SF网络效果不行，已测试
        features_DND = DND_features(g)
        with torch.no_grad():
            model_DND.eval()
            out = model_DND(features_DND.x.to(device), features_DND.edge_index.to(device))
            # print("Test output:", out)
            # print(out.shape)
        sorted_indices_DND = torch.argsort(out, descending=True)  # 获取out节点标签预测值从大到小排列的索引值
        DND_nodes = np.array(list(g.nodes))[sorted_indices_DND.tolist()]
        print(len(DND_nodes))



        # FINDER方法取节点
        print("FINDER")
        #idx2 = np.load('FINDER_selected_directed_test/' + network_name + '.npy').astype(int)
        idx2 = np.load('../Data/Otherdata/FD_selected/' + network_name + '.npy').astype(int)
        nodes = [str(item) for item in idx2]
        print(len(nodes))
        idx2 = nodes + list(set(g.nodes()) - set(nodes))


        # CoreHD方法取点
        print("CoreHD")
        corehd_g = g.copy()
        #CoreHD_nodes = corehd_disintegration(corehd_g)
        CoreHD_nodes  = corehd_disintegration(corehd_g)




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

        for i in range(len(back_nodes)):
            strong = max(nx.strongly_connected_components(g1), key=len)
            #strong_list.append(len(strong) / lscc0)
            strong_list.append(len(strong) / N)
            # print(i, len(strong) / N)
            edges = list(g1.in_edges(back_nodes[i])) + list(g1.out_edges(back_nodes[i]))
            g1.remove_edges_from(edges)


            strong_adapt_degree = max(nx.strongly_connected_components(g3), key=len)
            #strong_list_adapt_degree.append(len(strong_adapt_degree) / lscc0)
            strong_list_adapt_degree.append(len(strong_adapt_degree) / N)
            edges = list(g3.in_edges(adapt_degree[i])) + list(g3.out_edges(adapt_degree[i]))
            g3.remove_edges_from(edges)

            strong_degree = max(nx.strongly_connected_components(g4), key=len)
            #strong_list_degree.append(len(strong_degree) / lscc0)
            strong_list_degree.append(len(strong_degree) / N)
            edges = list(g4.in_edges(degree[i])) + list(g4.out_edges(degree[i]))
            g4.remove_edges_from(edges)

            strong_MS = max(nx.strongly_connected_components(g5), key=len)
            #strong_list_MS.append(len(strong_MS) / lscc0)
            strong_list_MS.append(len(strong_MS) / N)
            edges = list(g5.in_edges(minisum_nodes[i])) + list(g5.out_edges(minisum_nodes[i]))
            g5.remove_edges_from(edges)

            strong_pagerank = max(nx.strongly_connected_components(g6), key=len)
            #strong_list_pagerank.append(len(strong_pagerank) / lscc0)
            strong_list_pagerank.append(len(strong_pagerank) / N)
            edges = list(g6.in_edges(pagerank_nodes[i])) + list(g6.out_edges(pagerank_nodes[i]))
            g6.remove_edges_from(edges)

            strong_DND = max(nx.strongly_connected_components(g7), key=len)
            #strong_list_DND.append(len(strong_DND) / lscc0)
            strong_list_DND.append(len(strong_DND) / N)
            edges = list(g7.in_edges(str(DND_nodes[i]))) + list(g7.out_edges(str(DND_nodes[i])))
            g7.remove_edges_from(edges)

            strong_FINDER = max(nx.strongly_connected_components(g8), key=len)
            #strong_list_FINDER.append(len(strong_FINDER) / lscc0)
            strong_list_FINDER.append(len(strong_FINDER) / N)
            edges = list(g8.in_edges(str(idx2[i]))) + list(g8.out_edges(str(idx2[i])))
            g8.remove_edges_from(edges)

            strong_Core = max(nx.strongly_connected_components(g9), key=len)
            #strong_list_Core.append(len(strong_Core) / lscc0)
            strong_list_Core.append(len(strong_Core) / N)
            edges = list(g9.in_edges(CoreHD_nodes[i])) + list(g9.out_edges(CoreHD_nodes[i]))
            g9.remove_edges_from(edges)

        val = 1 / N

        strong_list += [val] * (N - i - 1)
        strong_list_degree += [val] * (N - i - 1)
        strong_list_adapt_degree += [val] * (N - i - 1)
        strong_list_MS += [val] * (N - i - 1)
        strong_list_pagerank += [val] * (N - i - 1)
        strong_list_DND += [val] * (N - i - 1)
        strong_list_FINDER += [val] * (N - i - 1)
        strong_list_Core += [val] * (N - i - 1)


        x = [_ / len(strong_list) for _ in range(len(strong_list))]
        show_n = int(len(strong_list) * 0.8)
        # 绘制瓦解曲线
        #plt.title(network_name, y=1.01, x=0.5, size=13)
        #plt.title(name, y=1.01, x=0.5, size=15)
        axesc[i0].plot(x[:show_n], strong_list[:show_n], color='#403990', lw=1.8, label="TAD")
        axesc[i0].plot(x[:show_n], strong_list_Core[:show_n], color="#888888", lw=1.2, label="CoreHD")
        axesc[i0].plot(x[:show_n], strong_list_pagerank[:show_n], color="#00FF00", lw=1.2, label="PageRk")
        axesc[i0].plot(x[:show_n], strong_list_MS[:show_n], color="#80A6E2", lw=1.2, label="MinSum")
        axesc[i0].plot(x[:show_n], strong_list_FINDER[:show_n], color="#FBDD85", lw=1.2, label="FINDER")
        axesc[i0].plot(x[:show_n], strong_list_DND[:show_n], color="#00FFFF", lw=1.2, label="DND")
        axesc[i0].plot(x[:show_n], strong_list_adapt_degree[:show_n], color="#F46F43", lw=1.2, label="HDA")
        axesc[i0].plot(x[:show_n], strong_list_degree[:show_n], color="#CF3D3E", lw=1.2, label="HD")
        #axes[0].xticks(fontsize=14)
        #axes[0].yticks(fontsize=14)
        #axesc[i0].set_ylim(-0.05, 1.05)
        axesc[i0].set_ylim(-0.05, 0.85)
        # fig.text(0.55, 0.02, 'Fraction of Nodes Removed', fontsize=16, ha='center')
        #fig.text(0.58, 0.01, 'Fraction of Nodes Removed', fontsize=16, ha='center')
        #fig.text(0.01, 0.55, 'GSCC', va='center', fontsize=16, rotation='vertical')
        #plt.legend(["STLD", 'MinSum', 'FINDER', 'PageRank', 'HDA', "HD", "DND", "CoreHD"], prop={ 'size': 10})

        # 标注采样点
        for idx, point in enumerate(sample_points):
            fraction_removed = point["fraction_removed"]
            gsc_size = point["gsc_size"]
            label = chr(66 + i0*4 + idx)  # 生成标签A, B, C, ...
            # 绘制短线
            axesc[i0].plot([fraction_removed, fraction_removed - 0.02],
                           [gsc_size, gsc_size - 0.01],
                           color='black', linestyle='-', linewidth=1)
            # 在短线末端标注字母
            axesc[i0].text(fraction_removed - 0.02, gsc_size - 0.01, label,
                           ha='right', va='top', fontsize=12, color='black')


        axesc[i0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        axesc[i0].set_title(f"{name}", fontsize=12)
        axesc[i0].set_xlabel("Fraction of Nodes Removed", fontsize=12)
        axesc[i0].set_ylabel("GSCC", fontsize=12)
        if network_name == "Neural_net_celegans_neural":
            axesc[i0].legend(loc='upper right', frameon=False)
        axesc[i0].text(-0.12, 1.1, chr(97+i0*4), transform=axesc[i0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.99, hspace=1.2, wspace=0.05)
    plt.show()


#draw_curve()

draw_heatmap()      #画真实网络热图。论文中的fig5

draw_curve_new_STLD()       #TAD方法瓦解曲线+多个拓扑图。论文中的fig6

