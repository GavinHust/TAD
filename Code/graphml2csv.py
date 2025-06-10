import networkx as nx
import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import lsmr
def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames

def get_levels(A):
    w_in = A.sum(axis=0) # 计算入度
    w_out = A.sum(axis=1)  # 计算出度
    u = w_in + w_out
    v = w_in - w_out
    Lambda = np.diag(u) - A - A.T
    h = lsmr(Lambda, v)[0]
    h = h - min(h)
    return h

def txt_to_graph(ba,n):

    G = nx.DiGraph()

    # matrix为邻接矩阵，以多维列表的形式存在

    nodes = range(n)
    G.add_nodes_from(nodes)

    for edges in ba:
        # edge = [int(x) for x in edges.split()[:-1]]
        edge = [int(x) for x in edges.split()]
        G.add_edge(edge[0]-1,edge[1]-1)

    return G

dir = "real_network/"

file_pre = "FoodWebs_reef"  # 文件以tes_开头
network_names = findfile(dir, file_pre)#[1:]
for network_name in network_names:
    # g=nx.read_graphml('all_graph/SF_50_2.4_7.0')
    g=nx.read_graphml('all_graph/'+file_pre)

    # f = open('real_network/' + network_name , 'r')
    # ba = f.readlines()
    # g = txt_to_graph(ba, 829)
    g.remove_edges_from(nx.selfloop_edges(g))
    # nx.write_graphml(g,'new_SF/'+file_pre)
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=np.float)
    h = get_levels(A)
    h = h.reshape((h.shape[0], 1))
    h_h = h.T - h
    h_h = np.where(h_h > 0, 1, h_h)  # 仅考虑back边的贡献度
    h_h = np.power((h_h - 1), 2)
    h_h_A = h_h * A

    up = h_h_A.sum(axis=1) + h_h_A.sum(axis=0)
    selected = np.argsort(up)[::-1]
    nodes = list(g.nodes)
    back_nodes = np.array(nodes)[selected]
    back_nodes = back_nodes.tolist() + list(set(g.nodes()) - set(back_nodes))
    data = pd.DataFrame(data=None, columns=['Source', 'Target', 'Type'])
    edges=list(g.edges)
    for edge in edges:
        data=data.append([{'Source':edge[0], 'Target':edge[1], 'Type':'Directed'}],ignore_index=True)
    data.to_csv("real_network/"+network_name+'.csv',index=False)