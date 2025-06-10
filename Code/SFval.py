import os
from scipy.optimize import curve_fit
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw


def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


def p_kin(kin, lamb_in, M_in, m_in):
    return ((kin + 1)**(1 - lamb_in) - kin**(1 - lamb_in)) / ((M_in + 1)**(1 - lamb_in) - m_in**(1 - lamb_in))
#dir = "new_SF/"
#dir = "bar_source/"
#dir = "F_networks/"
#dir = "biye_real_network/"
#dir = "DND_networks/new_networks/"
#file_pre = "SF_1000"  # 文件以tes_开头
#dir = "F_networks_new_SF/"
#dir = "F_networks_new/"
#dir = "SF_new/"
#dir = "biye_real_network/"
#dir = "pridect_SF_new/"
#dir = "F_networks_new_test_40sf/"
#dir = "F_networks_new_test_100sf/"
#file_pre = "SF_100000"  # 文件以tes_开头
#file_pre = "SF_1000"  # 文件以tes_开头
#dir = "SF_new_lamda2.8_10000/"
#dir = "SF_new_lamda_test/new_0.61/"
#dir = "F_ER_new/"
#dir = "F_ER_10000/"
#dir = "SF_FINDER/new3/"
#dir = "real_network/"
#dir = "biye_new_g/"
dir = "SF_new_lamda2-3.5/"

file_pre = "SF_1000_"  # 文件以tes_开头
filenames = findfile(dir, file_pre)
#filenames = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29', 'SF_1000_3.2_0.82_0.99_5.33']
#filenames = ["F_1000_0.2_0.075_9", "F_1000_0.3_0.25_9", "F_1000_0.4_0.45_9", "F_1000_0.5_0.75_9", "F_1000_0.6_1.17_9", "F_1000_0.7_2.05_9", "F_1000_0.8_5.998_9", "F_1000_1.0_29"]
#filenames = [ 'Genetic_net_p_aeruginosa','Genetic_net_m_tuberculosis','TRN-EC-RDB64', 'Neural_net_celegans_neural','Neural_rhesus_brain_1',
#                        'FoodWebs_Lough_Hyne', 'FoodWebs_little_rock',
#                     'p2p-Gnutella06', 'p2p-Gnutella08',  'Social-leader2Inter', 'Social_net_social_prison','Social_net_moreno_highschool', 'Freemans-EIES-1',
#                     'TexasPowerGrid','Trade_net_trade_food',
#                      "email-Eu-core","Wiki-Vote", "p2p-Gnutella09","p2p-Gnutella05","p2p-Gnutella04","p2p-Gnutella25","p2p-Gnutella24","p2p-Gnutella30"]
for file in filenames:
    filename = file
    #lambin = float(filename[8:11])
    #lambin = 2.988 #1.6 200,1.7 200, 1.8 600, 3.0 400,
    g=nx.read_graphml(dir+filename)         #读取graphhml形式储存的图
    print("节点数量",g.number_of_nodes())
    print("边数量",g.number_of_edges())
    print("平均度", 2* g.number_of_edges()/g.number_of_nodes())
    #g.remove_edges_from(nx.selfloop_edges(g))           #去掉指向自己的自环边

    strong_g = max(nx.strongly_connected_components(g), key=len)
    print("初始极大强连通分量", len(strong_g) / g.number_of_nodes())


    in_degrees = dict(g.in_degree())  # 获取所有节点的入度字典
    #in_degrees = {node: g.in_degree(node) + g.out_degree(node) for node in g.nodes()}
    #print(in_degrees)
    in_degree_vals = list(in_degrees.values())  # 获取所有入度值
    print(in_degree_vals)
    print(len(in_degree_vals))
    print(min(in_degree_vals))
    #in_degree_vals = [x for x in in_degree_vals if x!=0]
    #print(in_degree_vals)
    # 统计实际入度的频率
    unique_in_degrees, counts = np.unique(in_degree_vals, return_counts=True)
    #print(unique_in_degrees)
    #print(counts)
    #print(sum(counts))
    actual_p_kin = counts / sum(counts)  # 计算入度的概率分布        ,最小为1/sum(counts)，不能再小了，导致对数坐标系中x轴大时为横线，导致线性拟合出问题

    """
    out_degrees = dict(g.out_degree())  # 获取所有节点的出度字典
    out_degree_vals = list(out_degrees.values())  # 获取所有出度值
    print(out_degree_vals)
    out_degree_vals = [x for x in out_degree_vals if x!=0]
    #print(out_degree_vals)
    # 统计实际出度的频率
    unique_out_degrees, counts = np.unique(out_degree_vals, return_counts=True)
    #print(unique_out_degrees)
    actual_p_kout = counts / sum(counts)  # 计算出度的概率分布

    
    M_in = max(in_degree_vals)
    m_in = min(in_degree_vals)
    #print(M_in, m_in, lambin)
    theoretical_p_kin = [p_kin(kin, lambin, M_in, m_in) for kin in unique_in_degrees]
    """


    #求H，度平方的均值除以度均值的平方
    degrees = [d for n, d in g.degree()]
    mean_degree = np.mean(degrees)
    mean_degree_squared = np.mean([d ** 2 for d in degrees])
    H = mean_degree_squared / (mean_degree ** 2)


    #求实际的幂律指数
    M_in_1 = max(unique_in_degrees)
    m_in_1 = min(unique_in_degrees)
    initial_guess = [2.5]  # 假设lamb_in的初始猜测值为2.5
    popt, pcov = curve_fit(lambda kin, lamb_in: p_kin(kin, lamb_in, M_in_1, m_in_1), unique_in_degrees, actual_p_kin, p0=initial_guess)
    alpha = popt[0]
    #print("labda:",alpha)

    #线性回归拟合labda
    """
    # 对数变换
    log_probabilities = np.log(actual_p_kin)
    log_degrees = np.log(unique_in_degrees)
    # 线性回归
    slope, intercept, r_value, p_value, std_err = linregress(log_degrees, log_probabilities)
    # 斜率的负值就是幂律指数alpha
    alpha = -slope

    # 计算预测值
    predicted_probabilities = np.exp(intercept) * (unique_in_degrees ** slope)
    print(intercept)
    print(unique_in_degrees)
    print(alpha)
    print(predicted_probabilities)
    print(actual_p_kin)
    """

    predicted_probabilities = [p_kin(kin, popt, M_in_1, m_in_1) for kin in unique_in_degrees]
    # 计算误差
    errors = actual_p_kin - predicted_probabilities
    squared_errors = errors ** 2
    # 计算均方误差（MSE）
    mse = np.mean(squared_errors)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    pagerank_scores = list(nx.pagerank(g).values())
    #print(pagerank_scores)
    unique_pagerank, counts_pagerank = np.unique(pagerank_scores, return_counts=True)
    #print(sorted(unique_pagerank))
    actual_p_pagerank = counts_pagerank / sum(counts_pagerank)


    alpha = 2.2
    #print(M_in, m_in, lambin)
    theoretical_p_kin = [p_kin(kin, alpha, M_in_1, m_in_1) for kin in unique_in_degrees]

    # 拟合截断幂律分布
    fit = powerlaw.Fit(in_degree_vals,  min=min(in_degree_vals), discrete=True)
    print("alpha", fit.power_law.alpha)

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_in_degrees, actual_p_kin, color='blue', label='Actual In-degree Distribution', s=30, alpha=0.7)
    plt.plot(unique_in_degrees, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
    fit.power_law.plot_pdf(color='blue', ax=plt.gca())
    fit.power_law.plot_ccdf(color='green', label='CCDF', ax=plt.gca())

    plt.xlabel('$k_{in}$', fontsize=14)
    plt.ylabel('$p(k_{in})$', fontsize=14)
    plt.title(f'In-degree Distribution Comparison for {filename}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    # 在图上添加文本
    plt.text(1.01, 0.95,
             f'lamda: {alpha}\nH: {H}\nerror: {rmse}\n',
             ha='left', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.show()

    """
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_out_degrees, actual_p_kout, color='blue', label='Actual Out-degree Distribution', s=30, alpha=0.7)
    plt.xlabel('$k_out}$', fontsize=14)
    plt.ylabel('$p(k_{out})$', fontsize=14)
    plt.title(f'Out-degree Distribution Comparison for {filename}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    """

    """
    # 绘制
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_pagerank, actual_p_pagerank, color='blue', label='Actual PageRank Distribution', s=30, alpha=0.7)
    plt.xlabel('$pagerank$', fontsize=14)
    plt.ylabel('$p(pagerank)$', fontsize=14)
    plt.title(f'PageRank Distribution  for {filename}', fontsize=16)
    plt.legend()
    plt.grid(True)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    """