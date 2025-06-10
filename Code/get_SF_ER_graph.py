import networkx as nx
import os
from networkx.utils import powerlaw_sequence
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
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



def get_SF(N_ori,lamb, count):
    n, gamma = N_ori, lamb
    count_i =0
    while count_i <count:
        degree = powerlaw_sequence(n, gamma)
        out_degree = powerlaw_sequence(n, gamma)
        int_deg = np.array([round(deg) for deg in degree])
        # int_deg.sort()
        out_deg = np.array([round(deg) for deg in out_degree])
        diff=int_deg.sum()-out_deg.sum()
        #print(int_deg.sum())
        #print(out_deg.sum())
        #while(int_deg.sum()>0 and out_deg.sum()>0):
        if diff<0:
            out_deg[:-diff]-=1
        elif diff>0:
            #int_deg[:diff] -= 1
            out_deg[:diff] += 1
        print(int_deg.sum())
        print(out_deg.sum())
        #out_deg=[int_deg[i]+3 for i in range(N_ori//2)]+[int_deg[j]-3 for j in range(N_ori//2,N_ori//2*2)]+[int_deg[j] for j in range(N_ori//2*2,N_ori)]
        #out_deg=int_deg[::-1]
        #out_deg=np.hstack(((int_deg[:(N_ori-N_ori//10)]),out_deg[:N_ori//10]))
        #out_deg = random.sample(out_deg, n)
        try:
            g = nx.directed_havel_hakimi_graph(int_deg, out_deg)
            N = len(list(g.nodes))
            E = len(list(g.edges))

        except Exception as e:
            print("Error in creating the graph:", e)
            continue

        strong_g = max(nx.strongly_connected_components(g), key=len)
        LSCC_0 = len(strong_g) / g.number_of_nodes()
        if LSCC_0 < 0.5:
            continue
        # print(2 * E / N)

        network_name = 'SF_' + str(n) + '_' + str(gamma) + '_' + str(count_i) + '_' + str(round(2 * E / N, 2))
        # nx.write_graphml(g,'pridect_SF_new/'+network_name)
        # nx.write_graphml(g, 'SF_new_lamda2-3.5/' + network_name)
        nx.write_graphml(g, 'SF_FINDER/new4/' + network_name)
        count_i += 1
        print(count_i)


def get_SF_new(N_ori,lamb, count):
    n, gamma = N_ori, lamb
    count_i =0
    while count_i <count:
        degree = powerlaw_sequence(n, gamma)
        out_degree = powerlaw_sequence(n, gamma)
        int_deg = np.array([round(deg) for deg in degree])
        # int_deg.sort()
        out_deg = np.array([round(deg) for deg in out_degree])
        diff=int_deg.sum()-out_deg.sum()
        #print(int_deg.sum())
        #print(out_deg.sum())
        #while(int_deg.sum()>0 and out_deg.sum()>0):
        if diff<0:
            out_deg[:-diff]-=1
        elif diff>0:
            #int_deg[:diff] -= 1
            out_deg[:diff] += 1
        print(int_deg.sum())
        print(out_deg.sum())
        bili = max(round(5*N_ori/(int_deg.sum())), 1)
        #print(bili)
        int_deg = int_deg*bili
        out_deg = out_deg*bili
        print(int_deg.sum())
        print(out_deg.sum())
        #out_deg=[int_deg[i]+3 for i in range(N_ori//2)]+[int_deg[j]-3 for j in range(N_ori//2,N_ori//2*2)]+[int_deg[j] for j in range(N_ori//2*2,N_ori)]
        # out_deg=int_deg[::-1]
        #out_deg=np.hstack(((int_deg[:(N_ori-N_ori//10)]),out_deg[:N_ori//10]))
        #out_deg = random.sample(out_deg, n)
        try:
            g = nx.directed_havel_hakimi_graph(int_deg, out_deg)
            N = len(list(g.nodes))
            E = len(list(g.edges))
            #strong_g = max(nx.strongly_connected_components(g), key=len)
            #LSCC_0 = len(strong_g) / g.number_of_nodes()
            #if LSCC_0 <0.6:
            #    continue
            #print(2 * E / N)
        except Exception as e:
            # print("Error in creating the graph:", e)
            continue

        strong_g = max(nx.strongly_connected_components(g), key=len)
        LSCC_0 = len(strong_g) / g.number_of_nodes()
        if LSCC_0 < 0.5:
            continue

        network_name='SF_'+str(n)+'_'+str(gamma) +'_'+str(count_i) + "_" + str(round(2 * E / N, 2)) +"_1"
        #nx.write_graphml(g,'pridect_SF_new/'+network_name)
        nx.write_graphml(g, 'SF_FINDER/new3/' + network_name)
        count_i += 1

        print(count_i)




def get_ER(N_ori,avg_d,count):
    count_i = 0
    while count_i < count:
        g = nx.gnm_random_graph(N_ori, N_ori * avg_d / 2, directed=True)
        N = len(list(g.nodes))
        E = len(list(g.edges))
        #print(2 * E / N)
        #print(g.is_directed())
        network_name = 'ER_' + str(N_ori) + '_' + str(count_i) + "_" + str(round(2 * E / N, 2))
        strong_g = max(nx.strongly_connected_components(g), key=len)
        LSCC_0 = len(strong_g) / g.number_of_nodes()
        print(LSCC_0)
        if LSCC_0 < 0.2:
            continue
        #nx.write_graphml(g, 'SF_ER_FINDER/ER/' + network_name)
        nx.write_graphml(g, 'ER_new_lamda2-3.5/' + network_name)

        count_i += 1
        print(count_i)
        #return g, round(2 * E / N, 2)




#最开始的方式筛选LSCC0大于0.5，生成100、1000、10000
#D = [2.2, 2.4,2.6, 2.8]        #SF网络的幂律参数
D = [3, 6, 9, 12]
for D_ in D:
    #get_SF(10000, D_, 30)             #参数为节点数、幂律参数、数量
    #get_SF_new(500, D_, 30)           #平均度扩大3倍
    get_ER(1000, D_, 30)


