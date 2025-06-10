import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
import gc



# 生成满足目标 F 值的有向网络
def equation_show():

    G_0 = nx.DiGraph()
    nodes = list(range(5))  # 节点编号为 0 到 4
    G_0.add_nodes_from(nodes)
    # 手动指定边
    edges = [
        (1, 0), (1, 2), (1, 3), (2, 0), (2, 4)
    ]
    # 添加边到图中
    G_0.add_edges_from(edges)
    print(G_0)
    G = G_0.copy()
    # 转换为邻接矩阵
    A = nx.to_numpy_array(G)
    # 计算初始营养级和 F 值
    h = [0, 0.5, 2, 4, 5]
    print(h)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # 去掉子图外的边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 原始网络布局
    #pos = nx.kamada_kawai_layout(G_0)
    # 自定义布局，将节点编号小的放在下面
    pos = {0: (2, 0), 1: (1, 1), 2: (2, 2), 3: (0, 4), 4: (3, 3)}
    # 绘制节点
    nx.draw_networkx_nodes(G_0, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[(1, 3), (1, 2), (2, 4)], edge_color='blue', arrows=True, arrowsize=20,  ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[(1, 0), (2, 0)], edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    # 在节点旁边添加标注
    ax.text(pos[1][0], pos[1][1]-0.1, f"Node " + "$\mathit{i}$", fontsize=12, color='blue', ha='center')
    ax.text(pos[2][0], pos[2][1]-0.1, f"Node " + "$\mathit{j}$", fontsize=12, color='blue', ha='center')
    ax.text(pos[0][0], pos[0][1] - 0.3, f"h={0}", fontsize=12, color='red', ha='center')
    ax.text(pos[1][0], pos[1][1]-0.4, f"h={0.5}", fontsize=12, color='red', ha='center')
    ax.text(pos[2][0]+0.35, pos[2][1]-0.1, f"h={3}", fontsize=12, color='red', ha='center')
    ax.text(pos[3][0], pos[3][1]-0.5, f"h={5}", fontsize=12, color='red', ha='center')
    ax.text(pos[4][0], pos[4][1]-0.4, f"h={3.5}", fontsize=12, color='red', ha='center')

    #ax.set_title(f"A Subgraph Part Of A Hypothetical Network",  fontsize=12)

    # 显示整个图
    plt.tight_layout()
    plt.show()

equation_show()