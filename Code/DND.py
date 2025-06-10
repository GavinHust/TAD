import numpy as np
import random
import os
import itertools
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import glob
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import centrality


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
weights_history = []

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

    def forward(self, x, edge_index):               #pytorch训练和输入时会自动调用
        for layer in self.layers[:-1]:  # 前面的GAT层
            x = layer(x, edge_index)
        x = self.layers[-1](x, edge_index)  # 最后一层GAT
        x = self.mlp(x)
        return x.squeeze()





def compute_centrality_features(G):
    features = {
        "in_degree": [degree for node, degree in G.in_degree()],
        "out_degree": [degree for node, degree in G.out_degree()],
        "betweenness": list(centrality.betweenness_centrality(G).values()),
        "pagerank": list(nx.pagerank(G).values()),
    }
    return features


def find_minimum_disintegration_set(G, target_lscc_size):
    nodes = list(G.nodes)
    min_set_size = float('inf')
    optimal_sets = []
    lscc_size_min = float('inf')
    found = False
    for r in range(1, len(nodes) + 1):
        if found :  # 如果已经找到合适的集合，则跳出循环
            break
        if r>10:
            print("overtime", r)
            for comb in itertools.combinations(nodes, r):  # 迭代所有删除r个节点的组合
                H = G.copy()
                H.remove_nodes_from(comb)
                giant_component = max(nx.strongly_connected_components(H), key=len)
                lscc_size = len(giant_component)
                if lscc_size < lscc_size_min:
                    lscc_size_min = lscc_size
                    optimal_sets = [comb]
                elif lscc_size == lscc_size_min:
                    optimal_sets.append(comb)
            print("LSCC", lscc_size_min)
            node_labels = {node: 0 for node in nodes}
            for optimal_set in optimal_sets:
                for node in optimal_set:
                    node_labels[node] += 1 / len(optimal_sets)
            print(node_labels)
            return node_labels


        for comb in itertools.combinations(nodes, r):           #迭代所有删除r个节点的组合
            H = G.copy()
            H.remove_nodes_from(comb)
            giant_component = max(nx.strongly_connected_components(H), key=len)
            lscc_size = len(giant_component)
            if lscc_size < lscc_size_min:
                lscc_size_min = lscc_size
            if lscc_size <= target_lscc_size:
                if len(comb) < min_set_size:
                    min_set_size = len(comb)
                    min_set = comb
                    optimal_sets = [comb]
                    found = True
                    print(lscc_size)
                elif len(comb) == min_set_size:
                    optimal_sets.append(comb)
        print("r", r)
        print("lscc", lscc_size_min)

    node_labels = {node: 0 for node in nodes}
    for optimal_set in optimal_sets:
        for node in optimal_set:
            node_labels[node] += 1 / len(optimal_sets)
    print(node_labels)
    return node_labels




def load_data(file_pattern):
    files = glob.glob(file_pattern)
    data_list = []
    for file in files:
        with np.load(file, allow_pickle=True) as data:
            adj_matrix = data['adj_matrix']
            features_dict = data['features'].item()  # 转换为字典
            features = np.array([features_dict[key] for key in sorted(features_dict.keys())]).T
            labels = data['labels'].item()
            edge_index = np.array(adj_matrix.nonzero())
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor([labels[node] for node in labels.keys()], dtype=torch.float)
            print(x)
            print(y)
            print(edge_index)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
    return data_list



def train(model, data_loader, optimizer, loss_fn, epochs):
    model.train()
    losses = []  # 新增：用于记录每个epoch的损失
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # 模型的输入格式
            loss = loss_fn(out.squeeze(), data.y.to(device))           #计算模型前向传播forward的预测值，与标签值之间的损失
            loss.backward()                 #误差反向传播
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        layer_weights = {}
        for name, param in model.named_parameters():
            layer_weights[name] = param.detach().cpu().numpy().copy()  # 转换为CPU numpy数组，避免梯度影响
        weights_history.append(layer_weights)
    return losses


def save_model(model, optimizer, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)



def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


# 随机游走方法来选取子图
def random_walk_sampling(G, num_nodes):
    # 从随机起点开始
    start_node = random.choice(list(G.nodes))
    sampled_nodes = set([start_node])

    # 模拟随机游走直到收集到所需数量的节点
    current_node = start_node
    while len(sampled_nodes) < num_nodes:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:  # 若无邻居，则重新选择起点
            current_node = random.choice(list(G.nodes))
        else:
            # 随机选择一个邻居节点作为下一步
            next_node = random.choice(neighbors)
            sampled_nodes.add(next_node)
            current_node = next_node

    # 从采样的节点集构造子图
    return G.subgraph(sampled_nodes).copy()



def main():
    dir = "DND_fig/"           #用于训练合成网络
    output_dir = './DND_train_data/networkSFandER'
    file_pre = ""  # 文件以tes_开头
    mlp_dims = [100, 50, 1]  # 全连接层的神经元数量

    filenames = findfile(dir, file_pre)
    data_list = []
    """
    for i, file in enumerate(filenames):
        filename = file  # [:-4]
        
        g = nx.read_graphml(dir + filename)  # 读取graphhml形式储存的图
        g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边


        giant_component = max(nx.strongly_connected_components(g), key=len)
        lscc_size_0 = len(giant_component)
        if(lscc_size_0>1):
            if len(g.nodes) >= 25 :
                target_lscc_size = 1
            else:
                target_lscc_size = 1
            features = compute_centrality_features(g)
            #features = np.array([features_g[key] for key in sorted(features_g.keys())]).T           #介数、入度、出度、PageRank
            #print(features)
            labels = find_minimum_disintegration_set(g, target_lscc_size)
            #print(labels)


            edge_index = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
            #edge_index = np.array(adj_matrix.nonzero())
            #print(edge_index)


            #x = torch.tensor(features, dtype=torch.float)
            #y = torch.tensor([labels[node] for node in labels.keys()], dtype=torch.float)
            #edge_index = torch.tensor(edge_index, dtype=torch.long)

            save_path = "SF_" + str(file) + "_" + str(i)
            np.savez( os.path.join(output_dir, save_path), adj_matrix=edge_index, features=features, labels=labels)

            #data = Data(x=x, edge_index=edge_index, y=y)
            #data_list.append(data)
    """
    data_list = load_data(output_dir + "/SF_*.npz")
    print(data_list)
    data_loader = DataLoader(data_list, batch_size=1, shuffle=True)
    # model = GATNet(in_features=4, hidden_dim=8, out_features=8, num_heads=8)
    model = DeepGATNet(in_features=4, hidden_dims=[40, 30, 20, 10], out_features=mlp_dims[0],
                       heads_per_layer=[5, 5, 5, 5], mlp_dims=mlp_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000085)      #0.26，3000
    loss_fn = nn.BCELoss()
    #losses = train(model, data_loader, optimizer, loss_fn, epochs=1000)
    print(f"Using device: {device}")
    losses = train(model, data_loader, optimizer, loss_fn, epochs=200)
    #np.savez("weights_history.npz", weights_history=weights_history, allow_pickle=True)
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    save_model(model, optimizer, 'model_checkpoint_SF.pth')



if __name__ == "__main__":
    main()
    """
    # 加载 .npz 文件
    with np.load('weights_history.npz', allow_pickle=True) as data_file:
        # 通过名称访问数组
        loaded_array1 = data_file['weights_history']
    print(loaded_array1[0])
    print(loaded_array1[1])
    print(loaded_array1[2])
    print(loaded_array1[3])
    print(loaded_array1[4])"""
