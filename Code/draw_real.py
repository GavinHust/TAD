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
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

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



def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames
def draw_multi_bar():   #绘制多个网络的bar

    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter', 'Social_net_social_prison','Social_net_moreno_highschool', 'Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food', 'Average']
    fig, axes = plt.subplots(figsize=(15, 4))
    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.055, right=0.98)
    data = np.zeros(shape=(len(network_names), 9))
    for epoch in range(len(network_names) - 1):
        network_name = network_names[epoch]
        print('网络名称：', network_name)
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back = back / back[0]
        #backA = np.load('final_DN_result/' + network_name + '_backA.npy')
        #backA = backA / backA[0]
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
        DND = np.load('final_DN_result/' + network_name + '_DND.npy')      #DND2是最好
        DND = DND/DND[0]
        CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
        CoreHD = CoreHD/CoreHD[0]
        control = np.load('final_DN_result/' + network_name + '_control.npy')
        control = control/control[0]

        back_auc = round(back.sum() / len(back), 4)  # *100
        #backA_auc = round(backA.sum() / len(backA), 4)  # *100
        degree_auc = round(degree.sum() / len(back), 4)  # *100
        adpdegree_auc = round(adpdegree.sum() / len(back), 4)  # *100
        finder_auc = round(finder.sum() / len(back), 4)  # *100
        learn_auc = round(learn.sum() / len(back), 4)  # *100
        prank_auc = round(prank.sum() / len(back), 4)  # *100
        DND_auc = round(DND.sum() / len(back), 4)  # *100
        CoreHD_auc = round(CoreHD.sum() / len(back), 4)  # *100
        control_auc = round(control.sum() / len(back), 4)  # *100
        #print('back:', back_auc)
        #print('degree:', degree_auc)
        #print('adpdegree:', adpdegree_auc)
        #print('finder:', finder_auc)
        #print('learn:', learn_auc)
        #print('pagerank:', prank_auc)
        print('DND:', DND_auc)
        #print('CoreHD:', CoreHD_auc)
        #data[epoch, :] = np.array([back_auc, backA_auc, learn_auc, finder_auc, prank_auc, degree_auc, adpdegree_auc, DND_auc, CoreHD_auc])
        data[epoch, :] = np.array(
            [back_auc, CoreHD_auc, prank_auc, learn_auc, finder_auc, DND_auc, adpdegree_auc, degree_auc, control_auc])
    data[epoch + 1, :] = data.mean(axis=0)
    N = len(network_names)
    # 创建索引位置
    index = np.arange(N)
    width = 0.1  # 柱子的宽度
    #color = ['#403990', "#888888", "#80A6E2", "#FBDD85",  "#00FF00", "#F46F43", "#CF3D3E", "#00FFFF", "#000080"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    # 绘制第一组柱形图
    plt.bar(index, data[:,0], width, label='TAD',color=color[0])
    plt.bar(index + width, data[:, 1], width, label='CoreHD', color=color[1])
    # 绘制第二组柱形图，注意调整x的位置，以便与第一组柱形图分开
    plt.bar(index + 2*width, data[:,2], width, label='PageRank',color=color[2])
    plt.bar(index + 3*width, data[:,3], width, label='MinSum',color=color[3])
    plt.bar(index + 4 * width, data[:, 4], width, label='FINDER', color=color[4])
    plt.bar(index + 5*width, data[:,5], width, label='DND',color=color[5])
    plt.bar(index + 6*width, data[:,6], width, label='adpDegree',color=color[6])
    plt.bar(index + 7 * width, data[:, 7], width, label='Degree', color=color[7])
    #plt.bar(index + 8 * width, data[:, 8], width, label='Control', color=color[8])
    lamb = [ 'Gene01','Gene02', 'TRN', 'Neural01', 'Neural02',
             'Food\nWebs01', 'Food\nWebs02',
                     'p2p01', 'p2p02',  'Social01','Social02','Social03','Free-\nmans',
                    'Power\nGrid', 'Trade','Average']
    # 添加标题和标签
    # plt.xlabel('Categories',fontsize=13, fontname='Times New Roman')
    plt.ylabel('ANC',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    # plt.title('Multi-group Bar Chart')
    plt.xticks(index + width * 3, lamb)  # 设置x轴标签位置
    plt.legend(bbox_to_anchor=(0.55, 0.35),prop={ 'size': 11})  # 显示图例
    # plt.savefig('final_result/biye_' + 'multi_bar' + '.svg')
    # 显示图形
    plt.show()

def draw_curve():   #绘制单个网络的瓦解曲线
    """
    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'Metabolic-SC.s','Metabolic_net_TH','FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter','Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food']   #fig3中的真实曲线

    network_names = ['SF_500_3.6_0.74_0.25','SF_1000_2.6_0.33_0.18','SF_1000_3.6_0.91_0.9','SF_500_3.3_0.29_0.62']  #fig5中的瓦解曲线

    network_name = network_names[0]   #选择绘制哪一个网络
    """
    fig, axes = plt.subplots(figsize=(4.8, 3.2))
    fig.subplots_adjust(top=0.90, bottom=0.2, left=0.17, right=0.98 )
    #network_names=['TexasPowerGrid','SF_500_3.6_0.74_0.25','SF_1000_2.6_0.33_0.18','SF_1000_3.2_0.44_0.13','SF_1000_3.6_0.91_0.9','SF_500_3.3_0.29_0.62','Neural_rhesus_brain_2','FoodWebs_reef','SF_500_3.4_0.6_0.61','Trade_net_trade_basic']
    #name = network_names[0]

    network_name = 'p2p-Gnutella06'
    name = "p2p-Gnutella06"
    #network_name = 'FoodWebs_Lough_Hyne'
    #name = "FoodWebs_Lough_Hyne"

    back = np.load('final_DN_result/' + network_name + '_back.npy')
    #back = back / back[0]
    degree = np.load('final_DN_result/' + network_name + '_degree.npy')
    #degree = degree / degree[0]
    adpdegree = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
    #adpdegree = adpdegree / adpdegree[0]
    finder = np.load('final_DN_result/' + network_name + '_finder.npy')
    #finder = finder / finder[0]
    learn = np.load('final_DN_result/' + network_name + '_MS.npy')
    #learn = learn / learn[0]
    prank = np.load('final_DN_result/' + network_name + '_PR.npy')
    #prank = prank / prank[0]
    # rand = np.load('final_DN_result/' + network_name + '_rand.npy')
    DND = np.load('final_DN_result/' + network_name + '_DND.npy')
    #DND = DND / DND[0]
    CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
    #CoreHD = CoreHD / CoreHD[0]
    #control = np.load('final_DN_result/' + network_name + '_control.npy')
    #control = control / control[0]

    print(name,back.sum()/len(back))
    print(1-back.sum()/learn.sum(),1-back.sum()/finder.sum(),1-back.sum()/prank.sum(),1-back.sum()/adpdegree.sum(),1-back.sum()/degree.sum())

    x = [_ / len(back) for _ in range(len(back))]
    show_n=int(len(back)*0.6)
    # col.set_title(r'AvgD='+D[epoch],y=0.9)
    plt.title(network_name, y=1.01, x=0.5,size=13)
    plt.title(name, y=1.01, x=0.5,size=15)
    plt.plot(x[:show_n], back[:show_n], color='#403990', lw=1.8)
    plt.plot(x[:show_n], CoreHD[:show_n], color="#00FF00", lw=1.2)
    plt.plot(x[:show_n], finder[:show_n], color="#80A6E2", lw=1.2)
    plt.plot(x[:show_n], learn[:show_n], color="#FBDD85", lw=1.2)
    plt.plot(x[:show_n], prank[:show_n], color="#00FFFF", lw=1.2)
    plt.plot(x[:show_n], DND[:show_n], color="#F46F43", lw=1.2)
    plt.plot(x[:show_n], adpdegree[:show_n], color="#CF3D3E", lw=1.2)
    plt.plot(x[:show_n], degree[:show_n], color="#888888", lw=1.2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # fig.text(0.55, 0.02, 'Fraction of Nodes Removed', fontsize=16, ha='center')
    fig.text(0.58, 0.01, 'Fraction of Nodes Removed', fontsize=16, ha='center')
    fig.text(0.01, 0.55, 'GSCC',  va='center', fontsize=16, rotation='vertical')
    #plt.legend(["TAD", 'MinSum', 'FINDER', 'PageRank', 'HDA', "HD", "DND", "CoreHD"], prop={ 'size': 10})

    plt.show()

def draw_bar():   #绘制单个网络的bar
    unit_topics = ["TAD", "CoreHD", 'Finder', 'MinSum', 'PageRk', "DND", 'adpDegree', "Degree", "Control"]
    name = 'Metabolic-SC.s'
    #name = 'Metabolic_net_TH'
    #name = 'Freemans-EIES-1'
    fig, axes = plt.subplots( figsize=(16, 16))
    fig.subplots_adjust(top=0.9, bottom=0.25, left=0.2, right=0.92)

    color = ['#403990',  "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#888888", "#008080"]
    back = np.load('final_DN_result/' + name + '_back.npy')
    backA = np.load('final_DN_result/' + name + '_backA.npy')
    degree = np.load('final_DN_result/' + name + '_degree.npy')
    finder = np.load('final_DN_result/' + name + '_finder.npy')
    MS = np.load('final_DN_result/' + name + '_MS.npy')
    adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
    prank = np.load('final_DN_result/' + name + '_PR.npy')
    DND = np.load('final_DN_result/' + name + '_DND2.npy')
    CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
    #control = np.load('final_DN_result/' + name + '_control.npy')

    max_val = max(back)
    N=len(back)
    ba=back.sum() / (N * max_val)
    bAa = backA.sum() / (N * max_val)
    da=degree.sum() / (N * max_val)
    fa=finder.sum() / (N * max_val)
    ma=MS.sum() / (N * max_val)
    aa=adpDegree.sum() / (N * max_val)
    pa = prank.sum() / (N * max_val)
    DNDa = DND.sum() / (N * max_val)
    CoreHDa = CoreHD.sum() / (N * max_val)
    controla = control.sum() / (N * max_val)

            # rand_auc = round(rand.sum(), 2)
    std = [np.std(ba, ddof=1), np.std(CoreHDa, ddof=1), np.std(fa, ddof=1), np.std(ma, ddof=1), np.std(pa, ddof=1),
           np.std(DNDa, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
    # temp=[1,MS_auc/back_auc,finder_auc/back_auc,adpDegree_auc/back_auc,degree_auc/back_auc]
    temp = [round(ba,5), round(CoreHDa,5), round(fa,5), round(ma,5), round(pa,5), round(DNDa,5), round(aa,5), round(da,5), round(controla,5)]
    print(temp)
    print()
    for i in range(1,len(temp)):
        print(round(1-temp[0]/temp[i],2))
    bars = plt.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color, width=0.75)
    # 在每个柱子上方添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, round(height, 5), ha='center', va='bottom')
    plt.xticks(rotation=40)
    plt.title(name, y=1.0, x=0.5,fontsize=10)
    fig.text(0.04, 0.55, 'ANC', va='center', fontsize=10, rotation='vertical')
    # plt.savefig("final_result/bar_"+name+'.png')
    plt.show()



def draw_multi_bar_new():   #绘制多个网络的bar

    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter', 'Social_net_social_prison','Social_net_moreno_highschool',
                      'TexasPowerGrid','Trade_net_trade_food',
                      "Wiki-Vote",'Average']            #freemans节点数太少，email意义不大，p2p选两个就行
    fig, axes = plt.subplots(figsize=(18, 4))
    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.055, right=0.98)
    data = np.zeros(shape=(len(network_names), 8))
    for epoch in range(len(network_names) - 1):
        network_name = network_names[epoch]
        print('网络名称：', network_name)
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back = back / back[0]
        #backA = np.load('final_DN_result/' + network_name + '_backA.npy')
        #backA = backA / backA[0]
        degree = np.load('final_DN_result/' + network_name + '_degree.npy')
        degree = degree / degree[0]
        adpdegree = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        adpdegree = adpdegree / adpdegree[0]
        if network_name in ["Wiki-Vote"]:
            finder = np.array(0)
        else:
            finder = np.load('final_DN_result/' + network_name + '_finder.npy')
            finder = finder / finder[0]
        if network_name == "email-Eu-core":
            learn = np.array(0)
        else:
            learn = np.load('final_DN_result/' + network_name + '_MS.npy')
            learn = learn / learn[0]
        prank = np.load('final_DN_result/' + network_name + '_PR.npy')
        prank = prank / prank[0]
        # rand = np.load('final_DN_result/' + network_name + '_rand.npy')
        DND = np.load('final_DN_result/' + network_name + '_DND.npy')      #DND2是最好
        DND = DND/DND[0]
        CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
        CoreHD = CoreHD/CoreHD[0]
        #control = np.load('final_DN_result/' + network_name + '_control.npy')
        #control = control/control[0]

        back_auc = round(back.sum() / len(back), 4)  # *100
        #backA_auc = round(backA.sum() / len(backA), 4)  # *100
        degree_auc = round(degree.sum() / len(back), 4)  # *100
        adpdegree_auc = round(adpdegree.sum() / len(back), 4)  # *100
        finder_auc = round(finder.sum() / len(back), 4)  # *100
        learn_auc = round(learn.sum() / len(back), 4)  # *100
        prank_auc = round(prank.sum() / len(back), 4)  # *100
        DND_auc = round(DND.sum() / len(back), 4)  # *100
        CoreHD_auc = round(CoreHD.sum() / len(back), 4)  # *100
        #control_auc = round(control.sum() / len(back), 4)  # *100
        #print('back:', back_auc)
        #print('degree:', degree_auc)
        #print('adpdegree:', adpdegree_auc)
        #print('finder:', finder_auc)
        #print('learn:', learn_auc)
        #print('pagerank:', prank_auc)
        #print('DND:', DND_auc)
        #print('CoreHD:', CoreHD_auc)
        #data[epoch, :] = np.array([back_auc, CoreHD_auc, prank_auc, learn_auc, finder_auc, DND_auc, adpdegree_auc, degree_auc, control_auc])
        data[epoch, :] = np.array([back_auc, CoreHD_auc, prank_auc, learn_auc, finder_auc, DND_auc, adpdegree_auc, degree_auc])
    data[epoch + 1, :] = data.mean(axis=0)
    N = len(network_names)
    # 创建索引位置
    index = np.arange(N)
    width = 0.1  # 柱子的宽度
    #color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E"]
    # 绘制第一组柱形图
    plt.bar(index, data[:,0], width, label='TAD',color=color[0])
    #plt.bar(index+ width, data[:, 1], width, label='TADA', color=color[1])
    plt.bar(index + width, data[:, 1], width, label='CoreHD', color=color[1])
    # 绘制第二组柱形图，注意调整x的位置，以便与第一组柱形图分开
    plt.bar(index + 2*width, data[:,2], width, label='PageRank',color=color[2])
    plt.bar(index + 3*width, data[:,3], width, label='MinSum',color=color[3])
    plt.bar(index + 4 * width, data[:, 4], width, label='FINDER', color=color[4])
    plt.bar(index + 5*width, data[:,5], width, label='DND',color=color[5])
    plt.bar(index + 6*width, data[:,6], width, label='adpDegree',color=color[6])
    plt.bar(index + 7 * width, data[:, 7], width, label='Degree', color=color[7])
    #plt.bar(index + 8 * width, data[:, 8], width, label='Control', color=color[8])
    lamb = [ 'Gene01','Gene02', 'TRN', 'Neural01', 'Neural02',
             'Food\nWebs01', 'Food\nWebs02',
                     'p2p01', 'p2p02',  'Social01','Social02','Social03',
                    'Power\nGrid', 'Trade',"Wiki-Vote",'Average']
    # 添加标题和标签
    # plt.xlabel('Categories',fontsize=13, fontname='Times New Roman')
    plt.ylabel('ANC',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    # plt.title('Multi-group Bar Chart')
    plt.xticks(index + width * 3, lamb)  # 设置x轴标签位置
    plt.legend(bbox_to_anchor=(0.45, 0.35),prop={ 'size': 11})  # 显示图例
    # plt.savefig('final_result/biye_' + 'multi_bar' + '.svg')
    # 显示图形
    plt.show()


def draw_heatmap():
    network_names = [
        'FoodWebs_little_rock',"FoodWebs_Weddel_sea","subelj_cora.e",'p2p-Gnutella08',
        "Wiki-Vote",'p2p-Gnutella06',"ia-crime-moreno", "FoodWebs_reef",'Neural_net_celegans_neural',  "net_green_eggs",
        'Social-leader2Inter',
         "out.maayan-faa",'Neural_rhesus_brain_1',
         "Trade_net_trade_basic", 'Trade_net_trade_food',
          'Average'
    ]
    methods = ['TAD', 'CoreHD', 'PageRk', 'MinSum', 'FINDER', 'DND', 'HDA', 'HD']
    lamb = [  'Food\nWebs01','Food\nWebs02',"Scholarly01",
             'p2p08',"Wiki-Vote",'p2p06',"Crime", "Food\nWebs03", 'Neural01',"Language",
              'Social',"Infrastructure",
              'Neural02', 'Trade01','Trade02',
              'Average']
    # 数据矩阵
    data = np.zeros(shape=(len(methods), len(network_names)))

    # 填充数据矩阵
    for epoch in range(len(network_names) - 1):
        network_name = network_names[epoch]
        print('网络名称：', network_name)
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        #back = back / back[0]
        degree = np.load('final_DN_result/' + network_name + '_degree.npy')
        #degree = degree / degree[0]
        adpdegree = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        #adpdegree = adpdegree / adpdegree[0]
        finder = np.load('final_DN_result/' + network_name + '_finder.npy')
        #finder = finder / finder[0]
        learn = np.load('final_DN_result/' + network_name + '_MS.npy')
        #learn = learn / learn[0]
        prank = np.load('final_DN_result/' + network_name + '_PR.npy')
        #prank = prank / prank[0]
        DND = np.load('final_DN_result/' + network_name + '_DND.npy')
        #DND = DND / DND[0]
        CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
        #CoreHD = CoreHD / CoreHD[0]
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


    # 绘制热
    plt.figure(figsize=(12, 7))
    #ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues_r", cbar=False, xticklabels=lamb, yticklabels=methods,  annot_kws={"color": "black"})
    ax = sns.heatmap(data, annot=True, fmt=".3f", cmap=["#b3d9ff", "#0077b6"], cbar=False, xticklabels=lamb, yticklabels=methods,
                       linewidths=0, annot_kws={"color": "black", "fontsize": 12}, vmin=1, vmax=1, alpha=1)
    """# 添加列边框
    for i in range(len(network_names)+1):
        ax.axvline(x=i, color='black', linewidth=1)"""
    for i in range(data.shape[1]):  # 遍历每一列
        min_value = np.min(data[:, i])  # 找到当前列的最小值
        for j in range(data.shape[0]):  # 遍历当前列的每一行
            if data[j, i] == min_value:
                # 绘制红色矩形覆盖最小值区域
                rect = Rectangle((i, j), 1, 1, facecolor='#ff9999', edgecolor='none', zorder=1, alpha=1)
                ax.add_patch(rect)
    #plt.xlabel("Networks")
    #plt.ylabel("Methods")
    plt.title("AUC values of different dismantling methods in real networks", fontsize=14)
    plt.xticks(rotation=60, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)


    """polygon_coords = [
        (0 , 0.05),  # 左上角
        (11, 0.05),  # 左侧凹陷上部
        (11, 0 + 1),  # 左侧凹陷下部
        (12 , 0 + 1),  # 右侧凹陷下部
        (12, 0.05),  # 右侧凹陷上部

        (15 + 0.1, 0.05 ),  # 右上角

        (15 + 0.1, 0 + 1.2),  # 右下角


        (12, 0 + 1.2),  # 右侧凹陷上部
        (12, 1 + 1.2),  # 右侧凹陷下部
        (11, 1 + 1.2),  # 左侧凹陷下部
        (11, 0 + 1.2),  # 左侧凹陷上部
        (0, 0 + 1.2)  # 左下角
    ]
    polygon = Polygon(polygon_coords, closed=True, edgecolor='#CD5C5C', facecolor='none', linewidth=4)
    ax.add_patch(polygon)"""

    plt.tight_layout()
    plt.show()


# 绘制拓扑图并标出 LSCC 节点
def draw_topology_with_lsc(g, lsc_nodes, ax, method_name):
    print(len(lsc_nodes))

    # 加载背景图片
    #img_path = "foodweb.png"
    img_path = "celegans_neural.png"
    img = mpimg.imread(img_path)
    # 设置图片的显示范围（根据你的拓扑图坐标范围调整）
    img_extent = (-0.05, 1.05, -0.05, 1.05)  # (x_min, x_max, y_min, y_max)
    # 在当前子图中显示图片
    ax.imshow(img, extent=img_extent, aspect="auto", alpha=0.9)  # alpha 控制透明度

    pos = nx.random_layout(g, seed=40)
    node_colors = ['red' if node in lsc_nodes else 'gray' for node in g.nodes()]
    node_sizes = [20 if node in lsc_nodes else 10 for node in g.nodes()]

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    ax.set_title(f"{method_name}", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def draw_curve_new():   #绘制单个网络的瓦解曲线
    """
    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'Metabolic-SC.s','Metabolic_net_TH','FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter','Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food']   #fig3中的真实曲线
    network_name = network_names[0]   #选择绘制哪一个网络
    """

    dir = "biye_real_network/"
    #fig, axes = plt.subplots(2, 5, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})
    #fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(2, 6, figure=fig)
    # 合并第一列为一个大子图
    ax1 = fig.add_subplot(gs[:, :2])  # 占据第一列的所有行
    ax2 = fig.add_subplot(gs[0, 2])  # 第一行第二列
    ax3 = fig.add_subplot(gs[0, 3])  # 第一行第三列
    ax4 = fig.add_subplot(gs[0, 4])  # 第一行第四列
    ax5 = fig.add_subplot(gs[0, 5])  # 第一行第五列
    ax6 = fig.add_subplot(gs[1, 2])  # 第二行第二列
    ax7 = fig.add_subplot(gs[1, 3])  # 第二行第三列
    ax8 = fig.add_subplot(gs[1, 4])  # 第二行第四列
    ax9 = fig.add_subplot(gs[1, 5])  # 第二行第五列

    network_name = 'Neural_net_celegans_neural'
    #network_name = "p2p-Gnutella04"
    name = "NeuralNet_Celegans"
    #network_name = 'FoodWebs_Lough_Hyne'
    #name = "FoodWebs_Lough_Hyne"
    #network_name = 'Wiki-Vote'
    #name = "Wiki-Vote"
    #network_name = 'FoodWebs_little_rock'
    #name = "FoodWebs_little_rock"
    #network_name = 'Social_net_moreno_highschool'
    #name = 'Social_net_moreno_highschool'

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

    print(name)

    # 绘制拓扑图
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    nodes = list(g.nodes)
    N = len(nodes)
    print(N)
    edges = g.edges()
    a, b = zip(*edges)
    A = np.array(a)
    B = np.array(b)
    print(A)
    print(B)
    nodes_to_remove = int(N * 0.2)  # 需要移除的节点数量，为整个网络节点数量的10%

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
    back_nodes=back_nodes.tolist()+list(set(g.nodes)-set(back_nodes))
    back_nodes_remove = back_nodes[:nodes_to_remove]
    #print(len(set(back_nodes)))
    g_copy = g.copy()
    g_copy.remove_nodes_from(back_nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    print(lsc_nodes)
    draw_topology_with_lsc(g, lsc_nodes, ax2, "STLD")

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
    adapt_degree = adapt_degree + list(set(g.nodes()) - set(adapt_degree))
    adapt_degree_remove = adapt_degree[:nodes_to_remove]
    g_copy = g.copy()
    g_copy.remove_nodes_from(adapt_degree_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax3, "HDA")

    # 度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
    print("HD")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    D=A.sum(axis=0)+A.sum(axis=1)
    d = sorted(range(N), key=lambda k: D[k], reverse=True)
    degree=np.array(list(g.nodes))[d]
    degree=degree.tolist()+list(set(g.nodes())-set(degree))
    degree_remove = degree[:nodes_to_remove]
    g_copy = g.copy()
    g_copy.remove_nodes_from(degree_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax4, "HD")

    # MinSum方法取节点
    print("MiniSum")
    idx1 = np.load('SM_selected/' + network_name+'-output.npy').astype(int)
    #print(idx1)
    #idx2 = np.loadtxt('SM_selected/' + file + '.npy').astype(int)
    #print(idx2)
    minisum_nodes = np.array(list(g.nodes))[idx1]
    minisum_nodes = minisum_nodes.tolist() + list(set(g.nodes()) - set(minisum_nodes))
    minisum_nodes_remove = minisum_nodes[:nodes_to_remove]
    #print(minisum_nodes)
    g_copy = g.copy()
    g_copy.remove_nodes_from(minisum_nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax5, "MiniSum")

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
    pagerank_nodes = pagerank_nodes + list(set(g.nodes()) - set(pagerank_nodes))
    pagerank_nodes_remove = pagerank_nodes[:nodes_to_remove]
    g_copy = g.copy()
    g_copy.remove_nodes_from(pagerank_nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax6, "PageRank")

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
    DND_nodes = sorted_indices_DND.tolist()
    DND_nodes = [str(x) for x in DND_nodes]
    DND_nodes_remove = DND_nodes[:nodes_to_remove]
    g_copy = g.copy()
    g_copy.remove_nodes_from(DND_nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax7, "DND")


    # FINDER方法取节点
    print("FINDER")
    idx2 = np.load('FINDER_selected_directed_test/' + network_name + '.npy').astype(int)
    nodes = [str(item) for item in idx2]
    idx2 = nodes + list(set(g.nodes()) - set(nodes))
    idx2_remove = idx2[:nodes_to_remove]
    g_copy = g.copy()
    g_copy.remove_nodes_from(idx2_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax8, "FINDER")

    # CoreHD方法取点
    print("CoreHD")
    corehd_g = g.copy()
    #CoreHD_nodes = corehd_disintegration(corehd_g)
    CoreHD_nodes  = corehd_disintegration(corehd_g)
    CoreHD_nodes_remove = CoreHD_nodes[:nodes_to_remove]
    #print(CoreHD_nodes)
    g_copy = g.copy()
    g_copy.remove_nodes_from(CoreHD_nodes_remove)
    lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
    draw_topology_with_lsc(g, lsc_nodes, ax9, "CoreHD")



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
        strong_list.append(len(strong) / N)
        # print(i, len(strong) / N)
        edges = list(g1.in_edges(back_nodes[i])) + list(g1.out_edges(back_nodes[i]))
        g1.remove_edges_from(edges)


        strong_adapt_degree = max(nx.strongly_connected_components(g3), key=len)
        strong_list_adapt_degree.append(len(strong_adapt_degree) / N)
        edges = list(g3.in_edges(adapt_degree[i])) + list(g3.out_edges(adapt_degree[i]))
        g3.remove_edges_from(edges)

        strong_degree = max(nx.strongly_connected_components(g4), key=len)
        strong_list_degree.append(len(strong_degree) / N)
        edges = list(g4.in_edges(degree[i])) + list(g4.out_edges(degree[i]))
        g4.remove_edges_from(edges)

        strong_MS = max(nx.strongly_connected_components(g5), key=len)
        strong_list_MS.append(len(strong_MS) / N)
        edges = list(g5.in_edges(minisum_nodes[i])) + list(g5.out_edges(minisum_nodes[i]))
        g5.remove_edges_from(edges)

        strong_pagerank = max(nx.strongly_connected_components(g6), key=len)
        strong_list_pagerank.append(len(strong_pagerank) / N)
        edges = list(g6.in_edges(pagerank_nodes[i])) + list(g6.out_edges(pagerank_nodes[i]))
        g6.remove_edges_from(edges)

        strong_DND = max(nx.strongly_connected_components(g7), key=len)
        strong_list_DND.append(len(strong_DND) / N)
        edges = list(g7.in_edges(str(DND_nodes[i]))) + list(g7.out_edges(str(DND_nodes[i])))
        g7.remove_edges_from(edges)

        strong_FINDER = max(nx.strongly_connected_components(g8), key=len)
        strong_list_FINDER.append(len(strong_FINDER) / N)
        edges = list(g8.in_edges(str(idx2[i]))) + list(g8.out_edges(str(idx2[i])))
        g8.remove_edges_from(edges)

        strong_Core = max(nx.strongly_connected_components(g9), key=len)
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
    show_n = int(len(strong_list) * 0.6)
    # 绘制瓦解曲线
    #plt.title(network_name, y=1.01, x=0.5, size=13)
    #plt.title(name, y=1.01, x=0.5, size=15)
    ax1.plot(x[:show_n], strong_list[:show_n], color='#403990', lw=1.8, label="STLD")
    ax1.plot(x[:show_n], strong_list_Core[:show_n], color="#888888", lw=1.2, label="CoreHD")
    ax1.plot(x[:show_n], strong_list_pagerank[:show_n], color="#00FF00", lw=1.2, label="PageRk")
    ax1.plot(x[:show_n], strong_list_MS[:show_n], color="#80A6E2", lw=1.2, label="MinSum")
    ax1.plot(x[:show_n], strong_list_FINDER[:show_n], color="#FBDD85", lw=1.2, label="FINDER")
    ax1.plot(x[:show_n], strong_list_DND[:show_n], color="#00FFFF", lw=1.2, label="DND")
    ax1.plot(x[:show_n], strong_list_adapt_degree[:show_n], color="#F46F43", lw=1.2, label="HDA")
    ax1.plot(x[:show_n], strong_list_degree[:show_n], color="#CF3D3E", lw=1.2, label="HD")
    #axes[0].xticks(fontsize=14)
    #axes[0].yticks(fontsize=14)
    # fig.text(0.55, 0.02, 'Fraction of Nodes Removed', fontsize=16, ha='center')
    #fig.text(0.58, 0.01, 'Fraction of Nodes Removed', fontsize=16, ha='center')
    #fig.text(0.01, 0.55, 'GSCC', va='center', fontsize=16, rotation='vertical')
    #plt.legend(["STLD", 'MinSum', 'FINDER', 'PageRank', 'HDA', "HD", "DND", "CoreHD"], prop={ 'size': 10})
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax1.set_title(f"{name}", fontsize=15)
    ax1.set_xlabel("Fraction of Nodes Removed", fontsize=14)
    ax1.set_ylabel("GSCC", fontsize=14)
    ax1.legend(loc='upper right')




    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.07, right=0.99, hspace=0.2, wspace=0.1)
    plt.show()



# 绘制拓扑图并标出 LSCC 节点
def draw_topology_with_lsc_STLD(g, lsc_nodes, ax, method_name, i0, nodes_to_remove, gscc0 , label=None):
    print(len(lsc_nodes))

    pos = nx.random_layout(g, seed=40)
    node_colors = ['red' if node in lsc_nodes else 'gray' for node in g.nodes()]

    # 加载背景图片
    #img_path = "foodweb.png"
    if i0 ==0:
        img_path = "celegans_neural.png"
        # 调整 x 坐标范围到 (0, 0.5)
        pos = {node: (0.8 * pos[node][0], 0.2 + 0.8 * pos[node][1]) for node in pos}
        node_sizes = [2 if node in lsc_nodes else 1 for node in g.nodes()]
    else:
        img_path = "foodweb.png"
        # 调整 x 坐标范围到 (0, 0.5)
        pos = {node: (0.05 + 0.9 * pos[node][0], 0.05 + 0.9 * pos[node][1]) for node in pos}
        node_sizes = [20 if node in lsc_nodes else 15 for node in g.nodes()]
    img = mpimg.imread(img_path)
    # 设置图片的显示范围（根据你的拓扑图坐标范围调整）
    img_extent = (-0.05, 1.05, -0.05, 1.05)  # (x_min, x_max, y_min, y_max)
    # 在当前子图中显示图片
    ax.imshow(img, extent=img_extent, aspect="auto", alpha=1)  # alpha 控制透明度


    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #ax.set_title(f"GSCC_Size:{100*len(lsc_nodes)/len(g.nodes()):.1f}%\nRemoved Nodes:{nodes_to_remove*100:.0f}%", fontsize=10)
    ax.set_title(f"GSCC_Size:{100 * len(lsc_nodes) / gscc0:.1f}%\nRemoved Nodes:{nodes_to_remove * 100:.0f}%",
                 fontsize=10)
    #ax.text(0.5, -0.1, f"Removed Nodes:{nodes_to_remove * 100:.0f}%\nGSCC_Size:{len(lsc_nodes)}",fontsize=10, ha='center', va='top', transform=ax.transAxes)
    if label:
        ax.text(-0.1, 1.25, label, fontsize=12, ha='left', fontweight='bold', va='top', transform=ax.transAxes)


def draw_curve_new_STLD():   #绘制单个网络的瓦解曲线
    """
    network_names = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
                        'Metabolic-SC.s','Metabolic_net_TH','FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter','Freemans-EIES-1',
                      'TexasPowerGrid','Trade_net_trade_food']   #fig3中的真实曲线
    network_name = network_names[0]   #选择绘制哪一个网络
    """

    dir = "biye_real_network/"
    #fig, axes = plt.subplots(2, 5, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})
    #fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 7, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], height_ratios=[1, 0.5, 1.3])
    # 合并第一列为一个大子图
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])
    ax4 = fig.add_subplot(gs[:2, :3])

    ax5 = fig.add_subplot(gs[2, 4])
    ax6 = fig.add_subplot(gs[2, 5])
    ax7 = fig.add_subplot(gs[2, 6])
    ax8 = fig.add_subplot(gs[:2, 4:])

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
            nodes_to_remove_list = [int(N * 0.05),  int(N * 0.15), int(N * 0.25)] # 需要移除的节点数量，为整个网络节点数量的10%
            nodes_to_remove_list1 = [0.05, 0.15, 0.25]
        elif network_name == 'FoodWebs_reef':
            nodes_to_remove_list = [int(N * 0.05), int(N * 0.1), int(N * 0.15)]  # 需要移除的节点数量，为整个网络节点数量的10%
            nodes_to_remove_list1 = [0.05, 0.1, 0.15]

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
        back_nodes=back_nodes.tolist()+list(set(g.nodes)-set(back_nodes))
        for j,nodes_to_remove in enumerate(nodes_to_remove_list):
            back_nodes_remove = back_nodes[:nodes_to_remove]
            #print(len(set(back_nodes)))
            g_copy = g.copy()
            g_copy.remove_nodes_from(back_nodes_remove)
            lsc_nodes = max(nx.strongly_connected_components(g_copy), key=len)
            print(lsc_nodes)
            fraction_removed = nodes_to_remove / len(g.nodes())
            #gsc_size = len(lsc_nodes) / len(g.nodes())
            gsc_size = len(lsc_nodes) / lscc0
            sample_points.append({"fraction_removed": fraction_removed, "gsc_size": gsc_size})
            label = chr(67 + i0*3 + j)
            draw_topology_with_lsc_STLD(g, lsc_nodes, axes[i0][j], "STLD", i0, nodes_to_remove_list1[j], lscc0, label)

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
        idx1 = np.load('SM_selected/' + network_name+'-output.npy').astype(int)
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
                                            'model_checkpoint_SF.pth')  # model_checkpoint_SFandER在SF网络效果不行，已测试
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
        idx2 = np.load('FINDER_selected_directed/' + network_name + '.npy').astype(int)
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
            strong_list.append(len(strong) / lscc0)
            # print(i, len(strong) / N)
            edges = list(g1.in_edges(back_nodes[i])) + list(g1.out_edges(back_nodes[i]))
            g1.remove_edges_from(edges)


            strong_adapt_degree = max(nx.strongly_connected_components(g3), key=len)
            strong_list_adapt_degree.append(len(strong_adapt_degree) / lscc0)
            edges = list(g3.in_edges(adapt_degree[i])) + list(g3.out_edges(adapt_degree[i]))
            g3.remove_edges_from(edges)

            strong_degree = max(nx.strongly_connected_components(g4), key=len)
            strong_list_degree.append(len(strong_degree) / lscc0)
            edges = list(g4.in_edges(degree[i])) + list(g4.out_edges(degree[i]))
            g4.remove_edges_from(edges)

            strong_MS = max(nx.strongly_connected_components(g5), key=len)
            strong_list_MS.append(len(strong_MS) / lscc0)
            edges = list(g5.in_edges(minisum_nodes[i])) + list(g5.out_edges(minisum_nodes[i]))
            g5.remove_edges_from(edges)

            strong_pagerank = max(nx.strongly_connected_components(g6), key=len)
            strong_list_pagerank.append(len(strong_pagerank) / lscc0)
            edges = list(g6.in_edges(pagerank_nodes[i])) + list(g6.out_edges(pagerank_nodes[i]))
            g6.remove_edges_from(edges)

            strong_DND = max(nx.strongly_connected_components(g7), key=len)
            strong_list_DND.append(len(strong_DND) / lscc0)
            edges = list(g7.in_edges(str(DND_nodes[i]))) + list(g7.out_edges(str(DND_nodes[i])))
            g7.remove_edges_from(edges)

            strong_FINDER = max(nx.strongly_connected_components(g8), key=len)
            strong_list_FINDER.append(len(strong_FINDER) / lscc0)
            edges = list(g8.in_edges(str(idx2[i]))) + list(g8.out_edges(str(idx2[i])))
            g8.remove_edges_from(edges)

            strong_Core = max(nx.strongly_connected_components(g9), key=len)
            strong_list_Core.append(len(strong_Core) / lscc0)
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
        axesc[i0].set_ylim(-0.05, 1.05)
        # fig.text(0.55, 0.02, 'Fraction of Nodes Removed', fontsize=16, ha='center')
        #fig.text(0.58, 0.01, 'Fraction of Nodes Removed', fontsize=16, ha='center')
        #fig.text(0.01, 0.55, 'GSCC', va='center', fontsize=16, rotation='vertical')
        #plt.legend(["TAD", 'MinSum', 'FINDER', 'PageRank', 'HDA', "HD", "DND", "CoreHD"], prop={ 'size': 10})

        # 标注采样点
        for idx, point in enumerate(sample_points):
            fraction_removed = point["fraction_removed"]
            gsc_size = point["gsc_size"]
            label = chr(67 + i0*3 + idx)  # 生成标签A, B, C, ...
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
        axesc[i0].text(-0.14, 1.08, chr(65+i0), transform=axesc[i0].transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.07, right=0.99, hspace=1.1, wspace=0.18)
    plt.show()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#draw_bar()
#draw_curve()               #画单一真实网络的曲线图

#draw_multi_bar()            #画多个真实网络柱形图
#draw_multi_bar_new()

draw_heatmap()      #画真实网络热图

#draw_curve_new()       #各方法瓦解曲线+拓扑图
draw_curve_new_STLD()       #TAD方法瓦解曲线+多个拓扑图

