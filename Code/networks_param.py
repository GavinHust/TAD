import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lsmr
import networkx as nx
import os
import gc


# 得到营养级
def get_levels(A):
    w_in = A.sum(axis=0) # 计算入度
    w_out = A.sum(axis=1)  # 计算出度
    u = w_in + w_out
    v = w_in - w_out
    Lambda = np.diag(u) - A - A.T
    h = lsmr(Lambda, v)[0]
    h = h - min(h)          #保证营养级从0开始
    del Lambda,w_in,w_out,u,v
    gc.collect()
    return h

def calc_troph_incoh(A,h):
    F = 0
    idx=np.nonzero(A)
    for i in range(len(idx[0])):
        x=idx[0][i]
        y=idx[1][i]
        F = F + (h[y] - h[x] - 1) ** 2
    F = F / A.sum()
    del idx
    gc.collect()
    return F


def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


def p_kin(kin, lamb_in, M_in, m_in):
    return ((kin + 1)**(1 - lamb_in) - kin**(1 - lamb_in)) / ((M_in + 1)**(1 - lamb_in) - m_in**(1 - lamb_in))

dir = "biye_real_network/"
#dir = "F_networks/"
#dir = "biye_real_network_new_g/"
#dir = "bar_source/"
#dir = "all_graph/"
#dir = "SF_new_1/"
#dir = "SF_new_lamda2.8_10000/"
#dir = "SF_new_lamda2.8_10000_new/"
#dir = "F_networks_new/"
#dir = "F_networks_new_SF/"
#dir = "F_networks_new_test_40sf/"
#dir = "F_networks_new_test_100sf/"
#dir = "SF_new_lamda_test/0.1/"
#file_pre = "SF"  # 文件以tes_开头
#dir = "biye_real_network_test/"
#dir = "biye_new_g/"
#dir = "F_SF_new/"
#dir = "finder_real/"
file_pre = "TexasPowerGrid"  # 文件以tes_开头

filenames = findfile(dir, file_pre)
print(filenames)
print(len(filenames))

#filenames = ['F_1000_0.2_29', 'F_1000_0.5_29', 'F_1000_0.8_29', 'F_1000_1.0_29']
#filenames = [ "email-Eu-core","Wiki-Vote", "p2p-Gnutella09","p2p-Gnutella05","p2p-Gnutella04","p2p-Gnutella25","p2p-Gnutella24","p2p-Gnutella30"]
F_list = []
for file in filenames:
    filename = file
    #print(filename)
    g=nx.read_graphml(dir+filename)         #读取graphhml形式储存的图
    g.remove_edges_from(nx.selfloop_edges(g))           #去掉指向自己的自环边
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)



    # 获取节点数和边数
    node_count = g.number_of_nodes()  # 返回节点数
    edge_count = g.number_of_edges()  # 返回边数
    print("节点数:", node_count)
    print("边数:", edge_count)

    strong = max(nx.strongly_connected_components(g), key=len)
    """if (len(strong) / node_count) <0.1:
        print(filename)
        shutil.move(os.path.join(dir, filename), os.path.join("1/", filename))"""
    print("LSCC0", len(strong) / node_count)


    h = get_levels(A)
    F = calc_troph_incoh(A, h)
    print("F:", F)
    """
    F_list.append(F)
    #if 0.17 < round(F, 2) < 0.55:
    #    continue
    """
    """
    in_degrees = dict(g.in_degree())  # 获取所有节点的入度字典
    #print(in_degrees)
    in_degree_vals = list(in_degrees.values())  # 获取所有入度值
    print("入度序列",in_degree_vals)
    #print(len(in_degree_vals))

    out_degrees = dict(g.out_degree())  # 获取所有节点的入度字典
    #print(out_degrees)
    out_degree_vals = list(out_degrees.values())  # 获取所有入度值
    #print("出度序列",out_degree_vals)
    #print(len(out_degree_vals))

    d = dict(nx.degree(g))
    print("平均度：", sum(d.values()) / len(g.nodes))

    dot_product = np.dot(in_degree_vals, out_degree_vals)
    norm_in = np.linalg.norm(in_degree_vals)
    norm_out = np.linalg.norm(out_degree_vals)
    # 计算余弦相似性
    cosine_similarity = dot_product / (norm_in * norm_out)
    print("余弦相似性:", cosine_similarity)


    h = get_levels(A)
    F = calc_troph_incoh(A, h)
    print("F:", F)
    F_list.append(F)


    # 获取度分布
    nx.degree_histogram(g)  # 返回所有位于区间[0, dmax]的度值的频数列表
    print(max(d.values()))
    x = list(range(max(d.values()) + 1))
    # y = [i/len(G.nodes) for i in nx.degree_histogram(G)]
    y = [i / sum(nx.degree_histogram(g)) for i in nx.degree_histogram(g)]
    """
    """
    plt.bar(x, y, width=0.5, color="blue")
    plt.title(filename[:10])
    plt.xlabel("$k$")
    plt.ylabel("$p_k$")
    # 在图上添加文本
    plt.text(1.01, 0.95,
             f'N: {node_count}\nL: {edge_count}\nd: {sum(d.values()) / len(g.nodes):.3f}\nS: {cosine_similarity:.3f}\nF: {F:.1f}',
             ha='left', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.show()
    """

    #nx.write_graphml(g, 'SF_new_lamda2.8_10000_new/' + filename + "_" + str(round(F, 2)))
print(sorted(F_list))
print(len(sorted(F_list)))
# 绘制F值的频率分布图
plt.hist(F_list, bins=20, density=True, alpha=0.75, color='blue', edgecolor='black')
plt.title('f-F')
plt.xlabel('F')
plt.ylabel('f')
plt.show()

