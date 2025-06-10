import matplotlib.pyplot as plt
import numpy as np
import os
import math
import networkx as nx
import powerlaw
from scipy.special import kl_div
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.stats import ks_1samp
from scipy.integrate import cumtrapz


def findfile(directory, file_prefix):  # 获取directory路径下所有以file_prefix开头的文件
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


def draw_sf_er(type):
    dir = "bar_source/"  # 保存网络数据的文件夹

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    if type == 'SF':
        file_pre = "SF_1000"  # 文件以file_pre开头
        network_names = findfile(dir, file_pre)  # 获取满足要求的所有文件
        lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.5}$', r'$SF_{\lambda=2.8}$', r'$SF_{\lambda=3.2}$']
    elif type == 'ER':
        file_pre = "ER_1000"  # 文件以file_pre开头
        network_names = findfile(dir, file_pre)
        network_names = network_names[30:] + network_names[:30]  # 调整为参数从小到大的顺序
        lamb = [r'$ER_{\bar{D}=3}$', r'$ER_{\bar{D}=6}$', r'$ER_{\bar{D}=9}$', r'$ER_{\bar{D}=12}$']
    else:
        print('The \'type\' parameter can only be ER or SF')
        exit(0)

    num = 30  # 每组参数包含30个网络
    for j, col in enumerate(axes):

        back = np.zeros(1000)  # 初始化不同方法的平均GSCC曲线
        degree = np.zeros(1000)
        adpdegree = np.zeros(1000)
        finder = np.zeros(1000)
        learn = np.zeros(1000)
        pr = np.zeros(1000)
        dnd = np.zeros(1000)
        corehd = np.zeros(1000)
        control = np.zeros(1000)

        for epoch in range(num):  # 计算平均GSCC曲线
            network_name = network_names[j * num + epoch]
            temp = np.load('final_DN_result/' + network_name + '_back.npy')
            back += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_Core.npy')
            corehd += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_PR.npy')
            pr += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_degree.npy')
            degree += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
            adpdegree += temp / (num)
            # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
            if type == 'SF':
                temp = np.load('final_DN_result/' + network_name + '_FD.npy')
            elif type == 'ER':
                temp = np.load('final_DN_result/' + network_name + '_FD1.npy')  # FD1比FD好
            finder += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_MS.npy')
            learn += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_DND.npy')
            dnd += temp / (num)
            temp = np.load('final_DN_result/' + network_name + '_control.npy')
            control += temp / (num)

        x = [_ / len(back) for _ in range(len(back))]
        # col.set_title(r'AvgD='+D[epoch],y=0.9)
        col.set_title(lamb[j], y=0.8, x=0.5)
        col.plot(x, back, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
        col.plot(x, corehd, color="#888888", lw=1.2)
        col.plot(x, pr, color="#00FF00", lw=1.2)
        col.plot(x, learn, color="#80A6E2", lw=1.2)
        col.plot(x, finder, color="#FBDD85", lw=1.2)
        col.plot(x, dnd, color="#00FFFF", lw=1.2)
        col.plot(x, adpdegree, color="#F46F43", lw=1.2)
        col.plot(x, degree, color="#CF3D3E", lw=1.2)
        col.plot(x, control, color="#008000", lw=1.2)
        col.tick_params(labelsize=10)

    fig.text(0.5, 0.05, 'Fraction of Nodes Removed', fontsize=12, ha='center')
    fig.text(0.02, 0.5, 'GSCC', va='center', fontsize=12, rotation='vertical')
    plt.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 9},
               bbox_to_anchor=(0.9, 1.17), loc=1, ncol=9, borderaxespad=0)
    plt.show()


def draw_sf_er_single(type):
    dir = "bar_source/"  # 保存网络数据的文件夹

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    if type == 'SF':
        file_pre = "SF_1000"  # 文件以file_pre开头
        network_names = findfile(dir, file_pre)  # 获取满足要求的所有文件
        lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.5}$', r'$SF_{\lambda=2.8}$', r'$SF_{\lambda=3.2}$']
    elif type == 'ER':
        file_pre = "ER_1000"  # 文件以file_pre开头
        network_names = findfile(dir, file_pre)
        network_names = network_names[30:] + network_names[:30]  # 调整为参数从小到大的顺序
        lamb = [r'$ER_{\bar{D}=3}$', r'$ER_{\bar{D}=6}$', r'$ER_{\bar{D}=9}$', r'$ER_{\bar{D}=12}$']
    else:
        print('The \'type\' parameter can only be ER or SF')
        exit(0)

    for j, col in enumerate(axes):
        network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                         'SF_1000_3.2_0.82_0.99_5.33']
        network_name = network_names[j]
        temp = np.load('final_DN_result/' + network_name + '_back.npy')
        back = temp
        temp = np.load('final_DN_result/' + network_name + '_Core.npy')
        corehd = temp
        temp = np.load('final_DN_result/' + network_name + '_PR.npy')
        pr = temp
        temp = np.load('final_DN_result/' + network_name + '_degree.npy')
        degree = temp
        temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        adpdegree = temp
        # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
        temp = np.load('final_DN_result/' + network_name + '_FD.npy')  # 自己改为有向网络的FINDER             #怎么维度变成1001*1了
        finder = temp
        temp = np.load('final_DN_result/' + network_name + '_MS.npy')
        learn = temp
        temp = np.load('final_DN_result/' + network_name + '_DND.npy')
        dnd = temp
        temp = np.load('final_DN_result/' + network_name + '_control.npy')
        control = temp

        x = [_ / len(back) for _ in range(len(back))]
        # col.set_title(r'AvgD='+D[epoch],y=0.9)
        col.set_title(lamb[j], y=0.8, x=0.5)
        col.plot(x, back, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
        col.plot(x, corehd, color="#888888", lw=1.2)
        col.plot(x, pr, color="#00FF00", lw=1.2)
        col.plot(x, learn, color="#80A6E2", lw=1.2)
        col.plot(x, finder, color="#FBDD85", lw=1.2)
        col.plot(x, dnd, color="#00FFFF", lw=1.2)
        col.plot(x, adpdegree, color="#F46F43", lw=1.2)
        col.plot(x, degree, color="#CF3D3E", lw=1.2)
        col.plot(x, control, color="#008000", lw=1.2)
        col.tick_params(labelsize=10)

    fig.text(0.5, 0.05, 'Fraction of Nodes Removed', fontsize=12, ha='center')
    fig.text(0.02, 0.5, 'GSCC', va='center', fontsize=12, rotation='vertical')
    plt.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 9},
               bbox_to_anchor=(0.9, 1.17), loc=1, ncol=9, borderaxespad=0)
    plt.show()


def draw_F():
    dir = "F_networks/"  # 保存网络数据的文件夹
    file_pre = "F"
    network_names = findfile(dir, file_pre)
    lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.5}$', r'$SF_{\lambda=2.8}$', r'$SF_{\lambda=3.2}$']
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)

    for j, col in enumerate(axes):
        network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                         'SF_1000_3.2_0.82_0.99_5.33']

        network_name = network_names[j]
        back = np.load('final_F_result/' + network_name + '_backF.npy')
        back = [0 if math.isnan(x) else x for x in back]
        degree = np.load('final_F_result/' + network_name + '_degreeF.npy')
        degree = [0 if math.isnan(x) else x for x in degree]
        adpdegree = np.load('final_F_result/' + network_name + '_adpDegreeF.npy')
        adpdegree = [0 if math.isnan(x) else x for x in adpdegree]
        # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
        finder = np.load('final_F_result/' + network_name + '_FDF.npy')  # 自己改为有向网络的FINDER             #怎么维度变成1001*1了
        finder = [0 if math.isnan(x) else x for x in finder]
        learn = np.load('final_F_result/' + network_name + '_MSF.npy')
        learn = [0 if math.isnan(x) else x for x in learn]
        pr = np.load('final_F_result/' + network_name + '_PRF.npy')
        pr = [0 if math.isnan(x) else x for x in pr]
        dnd = np.load('final_F_result/' + network_name + '_DNDF.npy')
        dnd = [0 if math.isnan(x) else x for x in dnd]
        corehd = np.load('final_F_result/' + network_name + '_CoreF.npy')
        corehd = [0 if math.isnan(x) else x for x in corehd]
        control = np.load('final_F_result/' + network_name + '_controlF.npy')
        control = [0 if math.isnan(x) else x for x in control]

        x = [_ / len(back) for _ in range(len(back))]
        # col.set_title(r'AvgD='+D[epoch],y=0.9)
        col.set_title(lamb[j], y=0.8, x=0.5)
        col.plot(x, back, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
        col.plot(x, corehd, color="#888888", lw=1.2)
        col.plot(x, pr, color="#00FF00", lw=1.2)
        col.plot(x, learn, color="#80A6E2", lw=1.2)
        col.plot(x, finder, color="#FBDD85", lw=1.2)
        col.plot(x, dnd, color="#00FFFF", lw=1.2)
        col.plot(x, adpdegree, color="#F46F43", lw=1.2)
        col.plot(x, degree, color="#CF3D3E", lw=1.2)
        col.plot(x, control, color="#008000", lw=1.2)
        col.tick_params(labelsize=10)

    fig.text(0.5, 0.05, 'Fraction of Nodes Removed', fontsize=12, ha='center')
    fig.text(0.02, 0.5, 'F', va='center', fontsize=12, rotation='vertical')
    plt.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 9},
               bbox_to_anchor=(0.9, 1.17), loc=1, ncol=9, borderaxespad=0)
    plt.show()


def deta_GSCC():
    # dir = "biye_real_network/"
    # dir = "F_networks/"
    # dir = "bar_source/"
    # dir = "all_graph/"
    dir = "pridect_SF_new/"
    # dir = "SF_new/"
    # network_names = ['SF_1000_2.2_23.57', 'SF_1000_2.5_15.41', 'SF_1000_2.8_12.73','SF_1000_3.2_10.29']
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    network_names = ['SF_1000_2.2_0.18_0.73_11.01']
    network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    for j, ax in enumerate(axes):
        network_name = network_names[0]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 很关键，防止有多余的小数,1000个节点，保留3位，10000个节点保留4位，50000个节点保留5位
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_unique = back_unique * node_count
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        print(back_unique)
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + 0.2, 0.2)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        binned_p1 = []
        binned_p2 = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        if j == 1:
            """
            # 在子图上绘制
            #ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.set_title(network_name, fontsize=10)  # 子图标题
            ax.set_xlabel('deta_GSCC')  # 横坐标标签
            #ax.set_xscale('log')
            ax.set_ylabel('P')  # 纵坐标标签
            #ax.set_yscale('log')
            """
            # 在子图上绘制
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            # ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.set_title(network_name, fontsize=10)  # 子图标题
            ax.set_xlabel('deta_GSCC')  # 横坐标标签
            ax.set_xscale('log')
            ax.set_ylabel('P')  # 纵坐标标签
            ax.set_yscale('log')
        if j == 2:
            # 在子图上绘制
            # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.set_title(network_name, fontsize=10)  # 子图标题
            ax.set_xlabel('deta_GSCC')  # 横坐标标签
            ax.set_xscale('log')
            ax.set_ylabel('P')  # 纵坐标标签
            ax.set_yscale('log')
        if j == 0:
            x = [_ / len(back) for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)

    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()


def deta_GSCC_mult():
    # dir = "biye_real_network/"
    # dir = "F_networks/"
    dir = "bar_source/"
    # dir = "pridect_SF_new/"
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    # network_names = ['SF_1000_2.2_23.57', 'SF_1000_2.5_15.41', 'SF_1000_2.8_12.73','SF_1000_3.2_10.29']
    fig, axes = plt.subplots(9, 4, figsize=(12, 30))
    # fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97, hspace=0.5, wspace=0.3)
    box = 0.1
    for j, ax in enumerate(axes.flat[:4]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        print(back_deta)
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        # back_unique = back_unique*node_count
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")
        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))
        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)

        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")
        # 绘制拟合直线
        # ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('TAD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[4:8]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_degree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(binned_unique)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")
        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))
        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)

        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")
        # 绘制拟合直线
        # fit.power_law.plot_pdf(color='r', linestyle='--', linewidth=2, label=f'Power-law fit (α={alpha:.2f})', ax=ax)
        # ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        # ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('HD')  # 纵坐标标签
        ax.set_yscale('log')
        # ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[8:12]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('HDA')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[12:16]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_FD.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('FINDER')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[16:20]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_MS.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('MiniSum')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[20:24]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_PR.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('PageRank')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[24:28]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_DND.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('DND')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[28:32]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_Core.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('CoreHD')  # 纵坐标标签
        ax.set_yscale('log')
    for j, ax in enumerate(axes.flat[32:36]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = control = np.load('final_DN_result/' + network_name + '_control.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):

        # 在子图上绘制
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('Control')  # 纵坐标标签
        ax.set_yscale('log')
    # 显示图形
    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()


def deta_GSCC_mult_CDF():
    # dir = "biye_real_network/"
    # dir = "F_networks/"
    dir = "bar_source/"
    # dir = "pridect_SF_new/"
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    # network_names = ['SF_1000_2.2_23.57', 'SF_1000_2.5_15.41', 'SF_1000_2.8_12.73','SF_1000_3.2_10.29']
    fig, axes = plt.subplots(9, 4, figsize=(12, 30))
    # fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97, hspace=0.5, wspace=0.3)
    box = 0.1
    for j, ax in enumerate(axes.flat[:4]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        print(sorted_data)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        # pdf = back_counts / back_counts.sum()  # 计算出现的概率
        # back_unique = back_unique[1:]
        # pdf = pdf[1:]
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_title(network_name, fontsize=10)
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('TAD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[4:8]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_degree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('HD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[8:12]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('HDA')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[12:16]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_FD.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('FINDER')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[16:20]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_MS.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('MiniSum')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[20:24]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_PR.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('PageRank')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[24:28]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_DND.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('DND')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[28:32]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_Core.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('CoreHD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    for j, ax in enumerate(axes.flat[32:36]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = control = np.load('final_DN_result/' + network_name + '_control.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [data for data in back_deta if data != 0]
        # 排序数据
        sorted_data = np.sort(back_deta)
        # 计算累积分布
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # 绘制CDF
        ax.plot(sorted_data, cdf, label=f'{network_name}')
        ax.set_xscale('log')
        if j == 0:
            ax.set_ylabel('Control')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
    # 显示图形
    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()


def deta_GSCC_mult_new():
    # dir = "biye_real_network/"
    # dir = "F_networks/"
    dir = "bar_source/"
    # dir = "pridect_SF_new/"
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    # network_names = ['SF_1000_2.2_23.57', 'SF_1000_2.5_15.41', 'SF_1000_2.8_12.73','SF_1000_3.2_10.29']
    fig, axes = plt.subplots(9, 4, figsize=(15, 30))
    # fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.95, hspace=0.5, wspace=0.5)
    box = 0.1
    for j, ax in enumerate(axes.flat[:4]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('TAD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[4:8]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_degree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('HD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[8:12]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('HDA')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[12:16]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_FD.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('FINDER')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[16:20]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_MS.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('MiniSum')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[20:24]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_PR.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('PageRank')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[24:28]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_DND.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('DND')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[28:32]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_Core.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('CoreHD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    for j, ax in enumerate(axes.flat[32:36]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = control = np.load('final_DN_result/' + network_name + '_control.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 幂律拟合部分
        fit = powerlaw.Fit(back_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in back_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(back_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(back_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('Control')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
    # 显示图形
    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()


def deta_GSCC_mult_zhibiao():
    # dir = "biye_real_network/"
    # dir = "F_networks/"
    dir = "bar_source/"
    # dir = "pridect_SF_new/"
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    # network_names = ['SF_1000_2.2_23.57', 'SF_1000_2.5_15.41', 'SF_1000_2.8_12.73','SF_1000_3.2_10.29']
    fig, axes = plt.subplots(9, 4, figsize=(15, 30))
    # fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.95, hspace=0.5, wspace=0.5)
    box = 0.15

    la_all = {
        'SF_1000_2.2_0.2_0.55_9.48': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.5_0.49_0.9_7.13': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.8_0.6_0.87_6.29': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_3.2_0.82_0.99_5.33': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    M_all = {
        'SF_1000_2.2_0.2_0.55_9.48': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.5_0.49_0.9_7.13': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.8_0.6_0.87_6.29': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_3.2_0.82_0.99_5.33': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    D_all = {
        'SF_1000_2.2_0.2_0.55_9.48': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.5_0.49_0.9_7.13': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.8_0.6_0.87_6.29': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_3.2_0.82_0.99_5.33': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    MSE_all = {
        'SF_1000_2.2_0.2_0.55_9.48': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.5_0.49_0.9_7.13': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.8_0.6_0.87_6.29': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_3.2_0.82_0.99_5.33': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    KL_all = {
        'SF_1000_2.2_0.2_0.55_9.48': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.5_0.49_0.9_7.13': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_2.8_0.6_0.87_6.29': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SF_1000_3.2_0.82_0.99_5.33': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    for j, ax in enumerate(axes.flat[:4]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        print(back_deta)

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('TAD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][0] = alpha
        M_all[network_name][0] = xmax
        D_all[network_name][0] = D
        MSE_all[network_name][0] = mse
        KL_all[network_name][0] = kl_divergence

    for j, ax in enumerate(axes.flat[4:8]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_degree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('HD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][1] = alpha
        M_all[network_name][1] = xmax
        D_all[network_name][1] = D
        MSE_all[network_name][1] = mse
        KL_all[network_name][1] = kl_divergence

    for j, ax in enumerate(axes.flat[8:12]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('HDA')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][2] = alpha
        M_all[network_name][2] = xmax
        D_all[network_name][2] = D
        MSE_all[network_name][2] = mse
        KL_all[network_name][2] = kl_divergence

    for j, ax in enumerate(axes.flat[12:16]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_FD.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('FINDER')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][3] = alpha
        M_all[network_name][3] = xmax
        D_all[network_name][3] = D
        MSE_all[network_name][3] = mse
        KL_all[network_name][3] = kl_divergence

    for j, ax in enumerate(axes.flat[16:20]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_MS.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('MiniSum')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][4] = alpha
        M_all[network_name][4] = xmax
        D_all[network_name][4] = D
        MSE_all[network_name][4] = mse
        KL_all[network_name][4] = kl_divergence

    for j, ax in enumerate(axes.flat[20:24]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_PR.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率
        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('PageRank')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][5] = alpha
        M_all[network_name][5] = xmax
        D_all[network_name][5] = D
        MSE_all[network_name][5] = mse
        KL_all[network_name][5] = kl_divergence

    for j, ax in enumerate(axes.flat[24:28]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_DND.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('DND')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][6] = alpha
        M_all[network_name][6] = xmax
        D_all[network_name][6] = D
        MSE_all[network_name][6] = mse
        KL_all[network_name][6] = kl_divergence

    for j, ax in enumerate(axes.flat[28:32]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.load('final_DN_result/' + network_name + '_Core.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('CoreHD')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][7] = alpha
        M_all[network_name][7] = xmax
        D_all[network_name][7] = D
        MSE_all[network_name][7] = mse
        KL_all[network_name][7] = kl_divergence

    for j, ax in enumerate(axes.flat[32:36]):
        network_name = network_names[j]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = control = np.load('final_DN_result/' + network_name + '_control.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)
        back_deta = back_deta * node_count
        back_deta = [x for x in back_deta if x != 0]
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
        back_p = back_counts / back_counts.sum()  # 计算出现的概率

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        back_to_binned_mapping = {}
        binned_p = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                # 构建映射关系
                for unique_val in back_unique[bin_mask]:
                    back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
        binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
        print("back_unique:", back_unique)
        print("binned_unique:", binned_unique)
        print("back_to_binned_mapping:", back_to_binned_mapping)
        print("binned_p:", binned_p)
        print("binned_deta:", binned_deta)

        # 幂律拟合部分
        fit = powerlaw.Fit(binned_deta, xmin=1)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = max(back_deta)
        D = fit.power_law.D
        print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

        theoretical_p_kin = []
        for kin in binned_unique:
            theoretical_p_kin.append(
                ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

        # 计算均方误差（MSE）
        mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
        print(f"Mean Squared Error (MSE): {mse}")

        # 将实际概率和理论概率转换为数组
        binned_p_array = np.array(binned_p)
        theoretical_p_kin_array = np.array(theoretical_p_kin)
        # 计算KL散度
        kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
        print(f"Kullback-Leibler Divergence: {kl_divergence}")

        # 绘制拟合直线
        # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
        # 在子图上绘制
        # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
        # ax.plot(binned_unique, binned_p, label=f'{network_name}')
        # ax.set_title(network_name, fontsize=10)  # 子图标题
        ax.set_xlabel('deta_GSCC')  # 横坐标标签
        ax.set_xscale('log')
        ax.text(0.95, 0.98, f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                ha='left', va='top', transform=ax.transAxes, fontsize=9)
        if j == 0:
            ax.set_ylabel('Control')  # 纵坐标标签
        ax.set_yscale('log')
        ax.set_ylim(0.001, 1)
        la_all[network_name][8] = alpha
        M_all[network_name][8] = xmax
        D_all[network_name][8] = D
        MSE_all[network_name][8] = mse
        KL_all[network_name][8] = kl_divergence
    # 显示图形
    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()
    methods = ['TAD', 'HD', 'HDA', 'Finder', 'MiniSum', 'PageRank', "DND", "CoreHD", "Control"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.07, right=0.98, hspace=0.5)
    axes = axes.flatten()  # 将二维的axes数组展平为一维，方便后续遍历
    for i, ax in enumerate(axes):
        network = network_names[i]
        # ax.plot(methods, la_all[network], marker='o', linestyle='-', label="lamda")
        ax.bar(methods, M_all[network], color='#403990', label="Max")
        # ax.plot(methods, D_all[network], marker='o', linestyle='-', label="D")
        # ax.bar(methods, MSE_all[network], color='#403990', label="MSE")
        # ax.plot(methods, KL_all[network], marker='o', linestyle='-', label="KL")
        ax.set_title(network)
        ax.set_ylabel('Max', fontsize=12)
    # plt.suptitle('Difference of deta_GSCC in TAD')
    plt.tight_layout()
    plt.show()


def deta_GSCC_mult_zhibiao_mean():
    dir = "bar_source/"
    network_name_0 = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                      'SF_1000_3.2_0.82_0.99_5.33']
    network_names_pre = ['SF_1000_2.2_', 'SF_1000_2.5_', 'SF_1000_2.8_', 'SF_1000_3.2_']
    fig, axes = plt.subplots(9, 4, figsize=(15, 30))
    # fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.95, hspace=0.5, wspace=0.5)
    box = 0.15

    la_all = {
        'SF_1000_2.2_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.5_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.8_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_3.2_': [[], [], [], [], [], [], [], [], []]
    }
    M_all = {
        'SF_1000_2.2_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.5_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.8_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_3.2_': [[], [], [], [], [], [], [], [], []]
    }
    D_all = {
        'SF_1000_2.2_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.5_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.8_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_3.2_': [[], [], [], [], [], [], [], [], []]
    }
    MSE_all = {
        'SF_1000_2.2_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.5_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.8_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_3.2_': [[], [], [], [], [], [], [], [], []]
    }
    KL_all = {
        'SF_1000_2.2_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.5_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_2.8_': [[], [], [], [], [], [], [], [], []],
        'SF_1000_3.2_': [[], [], [], [], [], [], [], [], []]
    }

    for j, ax in enumerate(axes.flat[:4]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            print(file_pre, "len=" + str(len(network_names)))
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))

            back = np.load('final_DN_result/' + network_name + '_back.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率
            print(back_deta)

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('TAD')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][0].append(alpha)
            M_all[file_pre][0].append(xmax)
            D_all[file_pre][0].append(D)
            MSE_all[file_pre][0].append(mse)
            KL_all[file_pre][0].append(kl_divergence)

    for j, ax in enumerate(axes.flat[4:8]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_degree.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('HD')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][7].append(alpha)
            M_all[file_pre][7].append(xmax)
            D_all[file_pre][7].append(D)
            MSE_all[file_pre][7].append(mse)
            KL_all[file_pre][7].append(kl_divergence)

    for j, ax in enumerate(axes.flat[8:12]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率
            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('HDA')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][6].append(alpha)
            M_all[file_pre][6].append(xmax)
            D_all[file_pre][6].append(D)
            MSE_all[file_pre][6].append(mse)
            KL_all[file_pre][6].append(kl_divergence)

    for j, ax in enumerate(axes.flat[12:16]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_FD.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率
            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('FINDER')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][4].append(alpha)
            M_all[file_pre][4].append(xmax)
            D_all[file_pre][4].append(D)
            MSE_all[file_pre][4].append(mse)
            KL_all[file_pre][4].append(kl_divergence)

    for j, ax in enumerate(axes.flat[16:20]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_MS.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('MiniSum')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][3].append(alpha)
            M_all[file_pre][3].append(xmax)
            D_all[file_pre][3].append(D)
            MSE_all[file_pre][3].append(mse)
            KL_all[file_pre][3].append(kl_divergence)

    for j, ax in enumerate(axes.flat[20:24]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_PR.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率
            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('PageRank')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][2].append(alpha)
            M_all[file_pre][2].append(xmax)
            D_all[file_pre][2].append(D)
            MSE_all[file_pre][2].append(mse)
            KL_all[file_pre][2].append(kl_divergence)

    for j, ax in enumerate(axes.flat[24:28]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_DND.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('DND')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][5].append(alpha)
            M_all[file_pre][5].append(xmax)
            D_all[file_pre][5].append(D)
            MSE_all[file_pre][5].append(mse)
            KL_all[file_pre][5].append(kl_divergence)

    for j, ax in enumerate(axes.flat[28:32]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = np.load('final_DN_result/' + network_name + '_Core.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('CoreHD')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][1].append(alpha)
            M_all[file_pre][1].append(xmax)
            D_all[file_pre][1].append(D)
            MSE_all[file_pre][1].append(mse)
            KL_all[file_pre][1].append(kl_divergence)

    for j, ax in enumerate(axes.flat[32:36]):
        file_pre = network_names_pre[j]
        network_names = findfile(dir, file_pre)
        for network_name in network_names:
            g = nx.read_graphml(dir + network_name)
            node_count = g.number_of_nodes()
            lscc0_len = len(max(nx.strongly_connected_components(g), key=len))
            back = control = np.load('final_DN_result/' + network_name + '_control.npy')
            back_deta = -np.diff(back)
            back_deta = np.round(back_deta, 5)
            back_deta = back_deta * node_count
            back_deta = [x for x in back_deta if x != 0]
            # 计算变化量的频率分布
            back_unique, back_counts = np.unique(back_deta, return_counts=True)  # 统计变化量及其出现频率
            back_p = back_counts / back_counts.sum()  # 计算出现的概率

            # 将 back_unique 转换为对数坐标
            log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
            # 设置对数坐标上的分箱边界（等距选择 bin）
            # log_min = np.log10(back_unique.min())  # 对数坐标下限
            log_min = 0  # 对数坐标下限
            log_max = np.log10(back_unique.max())  # 对数坐标上限
            bin_edges_log = np.arange(log_min, log_max + box, box)  # 对数等距分箱
            bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
            # 将数据点分配到分箱
            bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
            print(bin_indices)
            # 初始化结果
            binned_unique = []
            back_to_binned_mapping = {}
            binned_p = []
            # 遍历每个分箱并计算均值和总和
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    # 分箱内求和
                    binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                    log_back_p = np.log10(back_p[bin_mask])
                    log_back_mean = log_back_p.mean()
                    back_p_mean = np.power(10, log_back_mean)
                    binned_p.append(back_p_mean)
                    # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
                    # 构建映射关系
                    for unique_val in back_unique[bin_mask]:
                        back_to_binned_mapping[unique_val] = back_unique[bin_mask].mean()  # 映射关系
            binned_deta = [back_to_binned_mapping.get(val, val) for val in back_deta]
            print("back_unique:", back_unique)
            print("binned_unique:", binned_unique)
            print("back_to_binned_mapping:", back_to_binned_mapping)
            print("binned_p:", binned_p)
            print("binned_deta:", binned_deta)

            # 幂律拟合部分
            fit = powerlaw.Fit(binned_deta, xmin=1)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = max(back_deta)
            D = fit.power_law.D
            print(f"Network: {network_name}, Alpha: {alpha}, Xmin: {xmin}, Xmax: {xmax},D: {D}")

            theoretical_p_kin = []
            for kin in binned_unique:
                theoretical_p_kin.append(
                    ((kin + 1) ** (1 - alpha) - kin ** (1 - alpha)) / ((xmax + 1) ** (1 - alpha) - xmin ** (1 - alpha)))

            # 计算均方误差（MSE）
            mse = np.mean((np.array(theoretical_p_kin) - np.array(binned_p)) ** 2)
            print(f"Mean Squared Error (MSE): {mse}")

            # 将实际概率和理论概率转换为数组
            binned_p_array = np.array(binned_p)
            theoretical_p_kin_array = np.array(theoretical_p_kin)
            # 计算KL散度
            kl_divergence = np.sum(kl_div(binned_p_array, theoretical_p_kin_array))
            print(f"Kullback-Leibler Divergence: {kl_divergence}")

            if network_name in network_name_0:
                # 绘制拟合直线
                # ax.plot(back_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                ax.plot(binned_unique, theoretical_p_kin, '#CF3D3E', label='Theoretical In-degree Distribution')
                # 在子图上绘制
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                # ax.plot(binned_unique, binned_p, label=f'{network_name}')
                # ax.set_title(network_name, fontsize=10)  # 子图标题
                ax.set_xlabel('deta_GSCC')  # 横坐标标签
                ax.set_xscale('log')
                ax.text(0.95, 0.98,
                        f'la: {alpha:.3f}\nM: {xmax:.3f}\nD: {D:.4f}\nMSE: {mse:.5f}\nKL: {kl_divergence:.4f}\n',
                        ha='left', va='top', transform=ax.transAxes, fontsize=9)
                if j == 0:
                    ax.set_ylabel('Control')  # 纵坐标标签
                ax.set_yscale('log')
                ax.set_ylim(0.001, 1)
            la_all[file_pre][8].append(alpha)
            M_all[file_pre][8].append(xmax)
            D_all[file_pre][8].append(D)
            MSE_all[file_pre][8].append(mse)
            KL_all[file_pre][8].append(kl_divergence)
    # 显示图形
    plt.suptitle('Difference of deta_GSCC in TAD')  # 整体标题
    plt.show()

    methods = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
    fig = plt.figure(figsize=(15, 6))  # 19，6
    gs = GridSpec(2, 3, figure=fig)
    # 合并第一列为一个大子图
    ax1 = fig.add_subplot(gs[0, 1])  # 第一行第二列
    ax2 = fig.add_subplot(gs[0, 2])  # 第一行第三列
    ax3 = fig.add_subplot(gs[1, 1])  # 第一行第四列
    ax4 = fig.add_subplot(gs[1, 2])  # 第一行第五列
    ax5 = fig.add_subplot(gs[:, 0:1])  # 占据第一列的所有行
    plt.subplots_adjust(top=0.98, bottom=0.5, left=0.02, right=0.98, wspace=0.2, hspace=0.5)  # 0.1
    # axes = axes.flatten()   # 将二维的axes数组展平为一维，方便后续遍历
    axes = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes):
        network_pre = network_names_pre[i]
        mse_values = MSE_all[network_pre]
        mse_mean = [np.mean(values) for values in mse_values]
        mse_se = [np.std(values) / np.sqrt(len(values)) for values in mse_values]

        """M_values = M_all[network_pre]
        M_mean = [np.mean(values) for values in M_values]
        #M_se = [(np.std(values)/1000) / np.sqrt(len(values)) for values in M_values]
        M_se = [(np.std(values, ddof=1) ) for values in M_values]"""
        M_values = M_all[network_pre]
        M_mean = [np.mean(values) / 1000 for values in M_values]
        M_se = [(np.std(values) / 1000) for values in M_values]
        # M_se = [(np.std(values, ddof=1)) for values in M_values]

        # ax.plot(methods, [x/30 for x in la_all[network_pre]], marker='o', linestyle='-', label="lamda")
        ax.bar(methods, M_mean, yerr=M_se, capsize=6,
               color=['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E",
                      "#008000"], edgecolor='black', label="Max")
        # ax.plot(methods, [x/30 for x in D_all[network_pre]], marker='o', linestyle='-', label="D")
        # ax.bar(methods, mse_mean, yerr=mse_se, capsize=10, color='#403990', edgecolor='black', label="MSE")
        # ax.plot(methods, [x/30 for x in KL_all[network_pre]], marker='o', linestyle='-', label="KL")

        network_title = network_pre[8:11]
        # ax.set_title(f"lamda={network_title}", fontsize=12)
        ax.set_title(r'$SF_{\lambda=' + network_title + '}$', fontsize=12)
        ax.set_xticklabels(methods, rotation=30, fontsize=11)
        # ax.set_ylim(bottom=0, top=230)
        ax.set_ylim(bottom=0, top=0.23)
        ax.set_ylabel('Max_avalanche', fontsize=12)
    # plt.suptitle('Difference of deta_GSCC in TAD')

    # 添加中间子图
    network_names = ['SF_1000_2.2_0.2_0.55_9.48', 'SF_1000_2.5_0.49_0.9_7.13', 'SF_1000_2.8_0.6_0.87_6.29',
                     'SF_1000_3.2_0.82_0.99_5.33']
    network_name = network_names[2]
    temp = np.load('final_DN_result/' + network_name + '_back.npy')
    back = temp
    temp = np.load('final_DN_result/' + network_name + '_Core.npy')
    corehd = temp
    temp = np.load('final_DN_result/' + network_name + '_PR.npy')
    pr = temp
    temp = np.load('final_DN_result/' + network_name + '_degree.npy')
    degree = temp
    temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
    adpdegree = temp
    # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
    temp = np.load('final_DN_result/' + network_name + '_FD.npy')  # 自己改为有向网络的FINDER             #怎么维度变成1001*1了
    finder = temp
    temp = np.load('final_DN_result/' + network_name + '_MS.npy')
    learn = temp
    temp = np.load('final_DN_result/' + network_name + '_DND.npy')
    dnd = temp
    temp = np.load('final_DN_result/' + network_name + '_control.npy')
    control = temp

    x = [_ / len(back) for _ in range(len(back))]

    # 定义一个函数来找到最大单步下降点并加粗
    def highlight_max_drop(ax, x, y, color, lw=1.8, highlight_lw=3.2):
        # 计算单步下降值
        drops = np.diff(y)
        max_drop_index = np.argmin(drops)  # 找到最大下降点的索引
        max_drop_x = x[max_drop_index]
        max_drop_y = y[max_drop_index]
        next_drop_y = y[max_drop_index + 1]
        # 绘制整条曲线
        ax.plot(x, y, color=color, lw=lw)
        # 加粗最大单步下降部分
        ax.plot([max_drop_x, x[max_drop_index + 1]], [max_drop_y, next_drop_y], color=color, lw=highlight_lw)

    highlight_max_drop(ax5, x, back, color='#403990', lw=1.8, highlight_lw=5)
    highlight_max_drop(ax5, x, corehd, color="#888888", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, pr, color="#00FF00", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, learn, color="#80A6E2", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, finder, color="#FBDD85", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, dnd, color="#00FFFF", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, adpdegree, color="#F46F43", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, degree, color="#CF3D3E", lw=1.2, highlight_lw=5)
    highlight_max_drop(ax5, x, control, color="#008000", lw=1.2, highlight_lw=5)

    ax5.tick_params(labelsize=10)
    ax5.set_ylabel("GSCC", fontsize=12)
    ax5.set_xlabel("Fraction of Nodes Removed", fontsize=12)
    # plt.subplots_adjust(top=0.98, bottom=0.08, left=0.07, right=0.98, hspace=0.15, wspace=1.2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    legend_handles = [
        ax5.plot([], [], color='#403990', lw=1.8)[0],
        ax5.plot([], [], color="#888888", lw=1.2)[0],
        ax5.plot([], [], color="#00FF00", lw=1.2)[0],
        ax5.plot([], [], color="#80A6E2", lw=1.2)[0],
        ax5.plot([], [], color="#FBDD85", lw=1.2)[0],
        ax5.plot([], [], color="#00FFFF", lw=1.2)[0],
        ax5.plot([], [], color="#F46F43", lw=1.2)[0],
        ax5.plot([], [], color="#CF3D3E", lw=1.2)[0],
        ax5.plot([], [], color="#008000", lw=1.2)[0]
    ]
    ax3.legend(legend_handles, ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"],
               prop={'size': 11},
               bbox_to_anchor=(-0.6, -0.32), loc='upper left', ncol=9, borderaxespad=0)
    ax5.text(-0.08, 1.05, 'A', transform=ax5.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax1.text(-0.1, 1.1, 'B', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax2.text(-0.1, 1.1, 'C', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.1, 1.1, 'D', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax4.text(-0.1, 1.1, 'E', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    plt.show()


def deta_GSCC_new():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    back = np.load('final_DN_result/' + network_name + '_back_new.npy')
    back = back * node_count
    print(back)
    print(len(back))
    # 从后向前遍历列表，直到遇到第一个大于1的值
    # 从后向前找到第一个大于1的值的位置
    # i = len(back) - 1
    # while i >= 0 and back[i] <= 1:
    #    i -= 1
    # back = back[:i + 1]

    print(back)
    print(len(back))
    back_deta = -np.diff(back)
    back_deta = np.round(back_deta, 5)  # 防止多余的小数
    fit = powerlaw.Fit(back_deta, xmin=1.0, discrete=True)
    print("alpha", fit.power_law.alpha)

    back_deta = back_deta[back_deta != 0]
    S1 = max(back_deta) + 1
    # epsilon = 1e-6
    epsilon = 0
    back_deta = back_deta + epsilon
    # 计算变化量的频率分布
    back_unique, back_counts = np.unique(back_deta, return_counts=True)
    print(back_counts)
    back_p = back_counts / back_counts.sum()
    print(back_unique)
    print(back_p)
    init_v = max(back_p)

    """
    # 定义带指数截断的幂律分布函数
    def truncated_power_law(S, t, S0, b):
        #epsilon = 1e-6
        #return S**(-t) * np.exp(S / S0) * b
        return np.where(S < S0, b * S**(-t), b * S0**(-t))
        #return (S + epsilon) ** (-t) * np.exp(-(S + epsilon) / S0) + b
    """

    def truncated_power_law(S, t, S0, b):
        # return np.where(S < S0, b * S ** (-t), b * S0 ** (-t))
        return np.where(S < 1, 0,  # 当 S < 0 时返回 0
                        np.where(S < S0, b * S ** (-t),  # 当 0 <= S < S0
                                 np.where(S < S1, b * S0 ** (-t), 0)))  # 当 S0 <= S < S1 或 S >= S1

    # 拟合带指数截断的幂律分布
    params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
    t, S0, b = params
    print(f"拟合参数: t = {t}, S0 = {S0}, S1 = {S1}, b = {b}")

    # 计算经验 CDF
    back_unique_sorted = np.sort(back_unique)  # 确保数据是排序的
    empirical_cdf = np.cumsum(back_p)  # 计算经验累积分布函数

    def truncated_power_law_cdf(S_values, t, S0, b):
        """计算离散情况下的理论 CDF"""
        S_values = np.array(S_values)  # 确保 S_values 是 NumPy 数组
        pdf_values = truncated_power_law(S_values, t, S0, b)  # 计算概率密度函数 (PDF)
        # pdf_values /= pdf_values.sum()  # 归一化，使其和为1
        return np.cumsum(pdf_values)  # 计算累计概率 (CDF)

    theoretical_cdf = truncated_power_law_cdf(back_unique_sorted, t, S0, b)
    print(empirical_cdf)
    print(theoretical_cdf)

    def theoretical_cdf_fun(x):
        idx = np.searchsorted(back_unique_sorted, x, side='right')
        print(np.where(idx == 0, 0, theoretical_cdf[idx - 1]))
        return np.where(idx == 0, 0, theoretical_cdf[idx - 1])

    # 计算 K-S 统计量和 p 值
    ks_statistic, p_value = ks_1samp(back_deta, theoretical_cdf_fun)
    print(back_unique)
    D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
    print(f"K-S Statistic: {ks_statistic}")
    print(f"D: {D_statistic}")
    print(f"P-Value: {p_value}")

    """
    # 计算 K-S 统计量（最大偏差）
    D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
    p_value = 1 - ksone.cdf(D_statistic, len(theoretical_cdf))  # 近似 p-value 计算

    # 输出检验结果
    print(f"K-S 统计量: {D_statistic}")
    print(f"p-value: {p_value}")


    # 通过 CDF 逆变换生成样本
    sample_size = 1000
    rand_uniform1 = np.random.rand(sample_size)
    rand_uniform2 = np.random.rand(sample_size)
    # 逆CDF 采样
    inv_cdf1 = interp1d(empirical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
    inv_cdf2 = interp1d(theoretical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
    sample1 = inv_cdf1(rand_uniform1)
    sample2 = inv_cdf2(rand_uniform2)
    #print(sample1)
    #print(sample2)
    # 进行 KS 检验
    ks_stat, p_value = ks_2samp(sample1, sample2)
    print(f"KS 统计量: {ks_stat}, p 值: {p_value}")

    # 逆变换采样生成样本数据
    def inverse_transform_sampling(cdf, values, n_samples):
        u = np.random.rand(n_samples)  # 生成均匀分布的随机数
        samples = np.interp(u, cdf, values)  # 通过插值找到对应的样本值
        return samples

    # 生成样本数据
    empirical_samples = inverse_transform_sampling(empirical_cdf, back_unique_sorted, 2000)
    theoretical_samples = inverse_transform_sampling(theoretical_cdf, back_unique_sorted, 2000)

    # 进行两样本K-S检验
    ks_statistic, p_value = ks_2samp(empirical_samples, theoretical_samples)
    print(f"K-S Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")
    """

    for j, ax in enumerate(axes):

        if j == 0:
            # x = [_ / len(back) for _ in range(len(back))]
            x = [_ / node_count for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_title(network_name, fontsize=10)
            # ax.set_xlim([0, 0.15])
        elif j == 1:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            # ax.set_xscale('log')
            ax.set_ylabel('P')
            # ax.set_yscale('log')
            ax.legend()
            # ax.set_xlim([0,1000])

            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f't = {t:.4f}\n\np-value = {p_value:.5f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
            plt.figure()
            fit.power_law.plot_pdf(color='green', ax=ax)
            # fit.power_law.plot_ccdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()

    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    node_count = g.number_of_nodes()
    back0 = np.load('final_DN_result/' + network_name + '_back.npy')
    p_list = []
    for i in range(1000 - 10):
        back = back0[:1000 - i]
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数
        back_deta = back_deta * node_count
        epsilon = 1e-6
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)
        # 拟合带指数截断的幂律分布
        params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
        t, S0, b = params
        print(f"拟合参数: t = {t}, S0 = {S0}, b = {b}")

        # 计算理论 CDF
        def truncated_power_law_cdf(S, t, S0, b):
            S_min, S_max = min(S), max(S)
            S_values = np.linspace(S_min, S_max, 1000)  # 生成连续范围
            P_values = truncated_power_law(S_values, t, S0, b)
            cdf_theoretical = cumtrapz(P_values, S_values, initial=0)  # 计算 CDF（累积分）
            # cdf_theoretical,  = quad(P_values, -np.inf, S_values)
            cdf_theoretical /= cdf_theoretical[-1]  # 归一化
            return S_values, cdf_theoretical

        # 计算样本 CDF
        S_values, cdf_theoretical = truncated_power_law_cdf(back_unique, t, S0, b)
        cdf_empirical = np.cumsum(back_p)  # 经验 CDF
        # 进行 K-S 检验
        ks_statistic, p_value = ks_1samp(back_unique, lambda x: np.interp(x, S_values, cdf_theoretical))
        p_list.append(p_value)
    p_list = p_list[::-1]
    print(f"p 值: {p_list}")
    indices = [i for i, value in enumerate(p_list) if value > 0.05]
    print(indices)  # 输出: [1, 3, 5]


def deta_GSCC_new_mult():
    # dir = "pridect_SF_new/"
    dir = "bar_source/"
    file_pre = "SF_1000_2.5_"
    filenames = findfile(dir, file_pre)
    network_names = filenames
    print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)

    # 定义带指数截断的幂律分布函数
    def truncated_power_law(S, t, S0, b):
        # epsilon = 1e-6
        return S ** (-t) * np.exp(-S / S0) * b
        # return (S + epsilon) ** (-t) * np.exp(-(S + epsilon) / S0) + b

    num = 30
    for j, ax in enumerate(axes):
        network_name = network_names[0]
        g = nx.read_graphml(dir + network_name)
        node_count = g.number_of_nodes()
        back = np.zeros(1000)
        for epoch in range(num):  # 计算平均GSCC曲线
            network_name = network_names[epoch]
            temp = np.load('final_DN_result/' + network_name + '_back.npy')
            back += temp / (num)
        # back = np.load('final_DN_result/' + network_name + '_back.npy')
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数

        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        back_unique = back_unique * node_count
        back_p = back_counts / back_counts.sum()
        epsilon = 1e-6
        back_unique = back_unique + epsilon
        print(back_unique)
        print(back_p)
        # 拟合带指数截断的幂律分布
        params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
        t, S0, b = params
        print(f"拟合参数: t = {t}, S0 = {S0}, b = {b}")

        # 计算理论 CDF
        def truncated_power_law_cdf(S, t, S0, b):
            S_min, S_max = min(S), max(S)
            S_values = np.linspace(S_min, S_max, 1000)  # 生成连续范围
            P_values = truncated_power_law(S_values, t, S0, b)
            cdf_theoretical = cumtrapz(P_values, S_values, initial=0)  # 计算 CDF（累积分）
            # cdf_theoretical,  = quad(P_values, -np.inf, S_values)
            cdf_theoretical /= cdf_theoretical[-1]  # 归一化
            return S_values, cdf_theoretical

        # 计算样本 CDF
        S_values, cdf_theoretical = truncated_power_law_cdf(back_unique, t, S0, b)
        cdf_empirical = np.cumsum(back_p)  # 经验 CDF

        # 进行 K-S 检验
        ks_statistic, p_value = ks_1samp(back_unique, lambda x: np.interp(x, S_values, cdf_theoretical))
        print(cdf_theoretical)
        print(cdf_empirical)
        # plt.figure(figsize=(6, 4))
        # plt.plot(back_unique, cdf_empirical, marker='o', linestyle='-', color='b', label='Empirical CDF')
        # plt.plot(S_values, cdf_theoretical, linestyle='--', color='r', label='Theoretical CDF (Fit)')

        print(f"K-S 统计量: {ks_statistic}, p 值: {p_value}")
        if j == 0:
            x = [_ / len(back) for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_xlim([0, 0.2])
            # ax.set_title(network_name, fontsize=10)
        elif j == 1:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

            # ax.set_title("30-SF", fontsize=10)
            ax.set_xlabel('deta_GSCC')
            # ax.set_xscale('log')
            ax.set_ylabel('P')
            # ax.set_yscale('log')
            # .legend()
            # ax.set_xlim([0,1000])
            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f't = {t:.4f}\n\np-value = {p_value:.5f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

            ax.set_title("30-SF", fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            # ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()


def deta_GSCC_new_bin():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    back = np.load('final_DN_result/' + network_name + '_back_new.npy')
    back = back * node_count
    print(back)
    print(len(back))
    # 从后向前遍历列表，直到遇到第一个大于1的值
    # 从后向前找到第一个大于1的值的位置
    # i = len(back) - 1
    # while i >= 0 and back[i] <= 1:
    #    i -= 1
    # back = back[:i + 1]

    print(back)
    print(len(back))
    back_deta = -np.diff(back)
    back_deta = np.round(back_deta, 5)  # 防止多余的小数
    print(len(back_deta))
    back_deta = back_deta[back_deta != 0]
    print(len(back_deta))
    S1 = max(back_deta) + 1
    # epsilon = 1e-6
    epsilon = 0
    back_deta = back_deta + epsilon
    # 计算变化量的频率分布
    back_unique, back_counts = np.unique(back_deta, return_counts=True)
    print(back_counts)
    back_p = back_counts / back_counts.sum()
    print(back_unique)
    print(back_p)

    # 将 back_unique 转换为对数坐标
    log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
    # 设置对数坐标上的分箱边界（等距选择 bin）
    # log_min = np.log10(back_unique.min())  # 对数坐标下限
    log_min = 0  # 对数坐标下限
    log_max = np.log10(back_unique.max())  # 对数坐标上限
    bin_edges_log = np.arange(log_min, log_max + 0.2, 0.2)  # 对数等距分箱
    bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
    # 将数据点分配到分箱
    bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
    print(bin_indices)
    # 初始化结果
    binned_unique = []
    binned_p = []
    binned_p1 = []
    binned_p2 = []
    # 遍历每个分箱并计算均值和总和
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            # 分箱内求和
            binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
            log_back_p = np.log10(back_p[bin_mask])
            log_back_mean = log_back_p.mean()
            back_p_mean = np.power(10, log_back_mean)
            binned_p.append(back_p_mean)
            # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
    print(binned_unique)
    print(binned_p)

    """
    # 定义带指数截断的幂律分布函数
    def truncated_power_law(S, t, S0, b):
        #epsilon = 1e-6
        #return S**(-t) * np.exp(S / S0) * b
        return np.where(S < S0, b * S**(-t), b * S0**(-t))
        #return (S + epsilon) ** (-t) * np.exp(-(S + epsilon) / S0) + b
    """

    def truncated_power_law(S, t, S0, b):
        # return np.where(S < S0, b * S ** (-t), b * S0 ** (-t))
        return np.where(S < min(binned_unique), 0,  # 当 S < 0 时返回 0
                        np.where(S < S0, b * S ** (-t),  # 当 0 <= S < S0
                                 np.where(S < S1, b * S0 ** (-t), 0)))  # 当 S0 <= S < S1 或 S >= S1

    # 拟合带指数截断的幂律分布
    params, covariance = curve_fit(truncated_power_law, binned_unique, binned_p, p0=[2.0, 10.0, 1.0])
    t, S0, b = params
    print(f"拟合参数: t = {t}, S0 = {S0}, S1 = {S1}, b = {b}")

    # 计算经验 CDF
    back_unique_sorted = np.sort(binned_unique)  # 确保数据是排序的
    empirical_cdf = np.cumsum(binned_p)  # 计算经验累积分布函数

    def truncated_power_law_cdf(S_values, t, S0, b):
        """计算离散情况下的理论 CDF"""
        S_values = np.array(S_values)  # 确保 S_values 是 NumPy 数组
        pdf_values = truncated_power_law(S_values, t, S0, b)  # 计算概率密度函数 (PDF)
        # pdf_values /= pdf_values.sum()  # 归一化，使其和为1
        return np.cumsum(pdf_values)  # 计算累计概率 (CDF)

    theoretical_cdf = truncated_power_law_cdf(back_unique_sorted, t, S0, b)
    print(empirical_cdf)
    print(theoretical_cdf)

    def theoretical_cdf_fun(x):
        idx = np.searchsorted(back_unique_sorted, x, side='right')
        print(np.where(idx == 0, 0, theoretical_cdf[idx - 1]))
        return np.where(idx == 0, 0, theoretical_cdf[idx - 1])

    # 计算 K-S 统计量和 p 值
    ks_statistic, p_value = ks_1samp(back_deta, theoretical_cdf_fun)
    print(back_unique)
    D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
    print(f"K-S Statistic: {ks_statistic}")
    print(f"D: {D_statistic}")
    print(f"P-Value: {p_value}")

    """
    # 计算 K-S 统计量（最大偏差）
    D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
    p_value = 1 - ksone.cdf(D_statistic, len(theoretical_cdf))  # 近似 p-value 计算

    # 输出检验结果
    print(f"K-S 统计量: {D_statistic}")
    print(f"p-value: {p_value}")


    # 通过 CDF 逆变换生成样本
    sample_size = 1000
    rand_uniform1 = np.random.rand(sample_size)
    rand_uniform2 = np.random.rand(sample_size)
    # 逆CDF 采样
    inv_cdf1 = interp1d(empirical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
    inv_cdf2 = interp1d(theoretical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
    sample1 = inv_cdf1(rand_uniform1)
    sample2 = inv_cdf2(rand_uniform2)
    #print(sample1)
    #print(sample2)
    # 进行 KS 检验
    ks_stat, p_value = ks_2samp(sample1, sample2)
    print(f"KS 统计量: {ks_stat}, p 值: {p_value}")

    # 逆变换采样生成样本数据
    def inverse_transform_sampling(cdf, values, n_samples):
        u = np.random.rand(n_samples)  # 生成均匀分布的随机数
        samples = np.interp(u, cdf, values)  # 通过插值找到对应的样本值
        return samples

    # 生成样本数据
    empirical_samples = inverse_transform_sampling(empirical_cdf, back_unique_sorted, 2000)
    theoretical_samples = inverse_transform_sampling(theoretical_cdf, back_unique_sorted, 2000)

    # 进行两样本K-S检验
    ks_statistic, p_value = ks_2samp(empirical_samples, theoretical_samples)
    print(f"K-S Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")
    """

    for j, ax in enumerate(axes):

        if j == 0:
            # x = [_ / len(back) for _ in range(len(back))]
            x = [_ / node_count for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_title(network_name, fontsize=10)
            # ax.set_xlim([0, 0.15])
        elif j == 1:
            # 绘制原始数据
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(binned_unique), max(binned_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            # ax.set_xscale('log')
            ax.set_ylabel('P')
            # ax.set_yscale('log')
            ax.legend()
            # ax.set_xlim([0,1000])

            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f't = {t:.4f}\n\np-value = {p_value:.5f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            # 绘制拟合曲线
            S_fit = np.linspace(min(binned_unique), max(binned_unique), 100)
            P_fit = truncated_power_law(S_fit, t, S0, b)
            ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()

    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    node_count = g.number_of_nodes()
    back0 = np.load('final_DN_result/' + network_name + '_back.npy')
    p_list = []
    for i in range(1000 - 10):
        back = back0[:1000 - i]
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数
        back_deta = back_deta * node_count
        epsilon = 1e-6
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)
        # 拟合带指数截断的幂律分布
        params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
        t, S0, b = params
        print(f"拟合参数: t = {t}, S0 = {S0}, b = {b}")

        # 计算理论 CDF
        def truncated_power_law_cdf(S, t, S0, b):
            S_min, S_max = min(S), max(S)
            S_values = np.linspace(S_min, S_max, 1000)  # 生成连续范围
            P_values = truncated_power_law(S_values, t, S0, b)
            cdf_theoretical = cumtrapz(P_values, S_values, initial=0)  # 计算 CDF（累积分）
            # cdf_theoretical,  = quad(P_values, -np.inf, S_values)
            cdf_theoretical /= cdf_theoretical[-1]  # 归一化
            return S_values, cdf_theoretical

        # 计算样本 CDF
        S_values, cdf_theoretical = truncated_power_law_cdf(back_unique, t, S0, b)
        cdf_empirical = np.cumsum(back_p)  # 经验 CDF
        # 进行 K-S 检验
        ks_statistic, p_value = ks_1samp(back_unique, lambda x: np.interp(x, S_values, cdf_theoretical))
        p_list.append(p_value)
    p_list = p_list[::-1]
    print(f"p 值: {p_list}")
    indices = [i for i, value in enumerate(p_list) if value > 0.05]
    print(indices)  # 输出: [1, 3, 5]


def deta_GSCC_new_mult2():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axess = plt.subplots(8, 3, figsize=(12, 15))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    method = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]

    def draw_fun(axes, path):
        back = np.load('final_DN_result/' + network_name + "_" + path + '_new.npy')
        back = back * node_count
        print(back)
        print(len(back))
        # 从后向前遍历列表，直到遇到第一个大于1的值
        # 从后向前找到第一个大于1的值的位置
        # i = len(back) - 1
        # while i >= 0 and back[i] <= 1:
        #    i -= 1
        # back = back[:i + 1]

        print(back)
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数

        back_deta = back_deta[back_deta != 0]
        S1 = max(back_deta) + 1
        # epsilon = 1e-6
        epsilon = 0
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)
        init_v = max(back_p)

        """
        # 定义带指数截断的幂律分布函数
        def truncated_power_law(S, t, S0, b):
            #epsilon = 1e-6
            #return S**(-t) * np.exp(S / S0) * b
            return np.where(S < S0, b * S**(-t), b * S0**(-t))
            #return (S + epsilon) ** (-t) * np.exp(-(S + epsilon) / S0) + b
        """

        def truncated_power_law(S, t, S0, b):
            # return np.where(S < S0, b * S ** (-t), b * S0 ** (-t))
            return np.where(S < 1, 0,  # 当 S < 0 时返回 0
                            np.where(S < S0, b * S ** (-t),  # 当 0 <= S < S0
                                     np.where(S < S1, b * S0 ** (-t), 0)))  # 当 S0 <= S < S1 或 S >= S1

        # 拟合带指数截断的幂律分布
        params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
        t, S0, b = params
        print(f"拟合参数: t = {t}, S0 = {S0}, S1 = {S1}, b = {b}")

        # 计算经验 CDF
        back_unique_sorted = np.sort(back_unique)  # 确保数据是排序的
        empirical_cdf = np.cumsum(back_p)  # 计算经验累积分布函数

        def truncated_power_law_cdf(S_values, t, S0, b):
            """计算离散情况下的理论 CDF"""
            S_values = np.array(S_values)  # 确保 S_values 是 NumPy 数组
            pdf_values = truncated_power_law(S_values, t, S0, b)  # 计算概率密度函数 (PDF)
            # pdf_values /= pdf_values.sum()  # 归一化，使其和为1
            return np.cumsum(pdf_values)  # 计算累计概率 (CDF)

        theoretical_cdf = truncated_power_law_cdf(back_unique_sorted, t, S0, b)
        print(empirical_cdf)
        print(theoretical_cdf)

        def theoretical_cdf_fun(x):
            idx = np.searchsorted(back_unique_sorted, x, side='right')
            print(np.where(idx == 0, 0, theoretical_cdf[idx - 1]))
            return np.where(idx == 0, 0, theoretical_cdf[idx - 1])

        # 计算 K-S 统计量和 p 值
        ks_statistic, p_value = ks_1samp([1, 233, 233], theoretical_cdf_fun)
        print(back_unique)
        D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        print(f"K-S Statistic: {ks_statistic}")
        print(f"D: {D_statistic}")
        print(f"P-Value: {p_value}")

        """
        # 计算 K-S 统计量（最大偏差）
        D_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        p_value = 1 - ksone.cdf(D_statistic, len(theoretical_cdf))  # 近似 p-value 计算

        # 输出检验结果
        print(f"K-S 统计量: {D_statistic}")
        print(f"p-value: {p_value}")


        # 通过 CDF 逆变换生成样本
        sample_size = 1000
        rand_uniform1 = np.random.rand(sample_size)
        rand_uniform2 = np.random.rand(sample_size)
        # 逆CDF 采样
        inv_cdf1 = interp1d(empirical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
        inv_cdf2 = interp1d(theoretical_cdf, back_unique_sorted, bounds_error=False, fill_value="extrapolate")
        sample1 = inv_cdf1(rand_uniform1)
        sample2 = inv_cdf2(rand_uniform2)
        #print(sample1)
        #print(sample2)
        # 进行 KS 检验
        ks_stat, p_value = ks_2samp(sample1, sample2)
        print(f"KS 统计量: {ks_stat}, p 值: {p_value}")

        # 逆变换采样生成样本数据
        def inverse_transform_sampling(cdf, values, n_samples):
            u = np.random.rand(n_samples)  # 生成均匀分布的随机数
            samples = np.interp(u, cdf, values)  # 通过插值找到对应的样本值
            return samples

        # 生成样本数据
        empirical_samples = inverse_transform_sampling(empirical_cdf, back_unique_sorted, 2000)
        theoretical_samples = inverse_transform_sampling(theoretical_cdf, back_unique_sorted, 2000)

        # 进行两样本K-S检验
        ks_statistic, p_value = ks_2samp(empirical_samples, theoretical_samples)
        print(f"K-S Statistic: {ks_statistic}")
        print(f"P-Value: {p_value}")
        """

        for j, ax in enumerate(axes):

            if j == 0:
                # x = [_ / len(back) for _ in range(len(back))]
                x = [_ / node_count for _ in range(len(back))]
                # col.set_title(r'AvgD='+D[epoch],y=0.9)
                ax.set_title("TAD", y=0.8, x=0.5)
                ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
                ax.tick_params(labelsize=10)
                ax.set_title(network_name, fontsize=10)
                # ax.set_xlim([0, 0.15])
            elif j == 1:
                # 绘制原始数据
                ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

                # 绘制拟合曲线
                S_fit = np.linspace(min(back_unique), max(back_unique), 100)
                P_fit = truncated_power_law(S_fit, t, S0, b)
                ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

                ax.set_title(network_name, fontsize=10)
                ax.set_xlabel('deta_GSCC')
                # ax.set_xscale('log')
                ax.set_ylabel('P')
                # ax.set_yscale('log')
                ax.legend()
                # ax.set_xlim([0,1000])

                # 在图上显示 t 的值
                ax.text(0.6, 0.4, f't = {t:.4f}\n\np-value = {p_value:.5f}', transform=ax.transAxes, fontsize=12,
                        ha='center', va='center')
            else:
                # 绘制原始数据
                ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

                # 绘制拟合曲线
                S_fit = np.linspace(min(back_unique), max(back_unique), 100)
                P_fit = truncated_power_law(S_fit, t, S0, b)
                ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')

                ax.set_title(network_name, fontsize=10)
                ax.set_xlabel('deta_GSCC')
                ax.set_xscale('log')
                ax.set_ylabel('P')
                ax.set_yscale('log')
                ax.legend()
                ax.set_xlim([0, 1000])

    for i, method_name in enumerate(method):
        print(i)
        print(method_name)
        draw_fun(axess[i], method_name)
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()

    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    node_count = g.number_of_nodes()
    back0 = np.load('final_DN_result/' + network_name + '_back.npy')
    p_list = []
    for i in range(1000 - 10):
        back = back0[:1000 - i]
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数
        back_deta = back_deta * node_count
        epsilon = 1e-6
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)
        # 拟合带指数截断的幂律分布
        params, covariance = curve_fit(truncated_power_law, back_unique, back_p, p0=[2.0, 10.0, 1.0])
        t, S0, b = params
        print(f"拟合参数: t = {t}, S0 = {S0}, b = {b}")

        # 计算理论 CDF
        def truncated_power_law_cdf(S, t, S0, b):
            S_min, S_max = min(S), max(S)
            S_values = np.linspace(S_min, S_max, 1000)  # 生成连续范围
            P_values = truncated_power_law(S_values, t, S0, b)
            cdf_theoretical = cumtrapz(P_values, S_values, initial=0)  # 计算 CDF（累积分）
            # cdf_theoretical,  = quad(P_values, -np.inf, S_values)
            cdf_theoretical /= cdf_theoretical[-1]  # 归一化
            return S_values, cdf_theoretical

        # 计算样本 CDF
        S_values, cdf_theoretical = truncated_power_law_cdf(back_unique, t, S0, b)
        cdf_empirical = np.cumsum(back_p)  # 经验 CDF
        # 进行 K-S 检验
        ks_statistic, p_value = ks_1samp(back_unique, lambda x: np.interp(x, S_values, cdf_theoretical))
        p_list.append(p_value)
    p_list = p_list[::-1]
    print(f"p 值: {p_list}")
    indices = [i for i, value in enumerate(p_list) if value > 0.05]
    print(indices)  # 输出: [1, 3, 5]


def deta_GSCC_new_1():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    back = np.load('final_DN_result/' + network_name + '_back_new.npy')
    back = back * node_count
    print(back)
    print(len(back))
    # 从后向前遍历列表，直到遇到第一个大于1的值
    # 从后向前找到第一个大于1的值的位置
    # i = len(back) - 1
    # while i >= 0 and back[i] <= 1:
    #    i -= 1
    # back = back[:i + 1]

    print(back)
    print(len(back))
    back_deta = -np.diff(back)
    back_deta = np.round(back_deta, 5)  # 防止多余的小数

    back_deta = back_deta[back_deta != 0]
    # epsilon = 1e-6
    epsilon = 0
    back_deta = back_deta + epsilon
    # 计算变化量的频率分布
    back_unique, back_counts = np.unique(back_deta, return_counts=True)
    print(back_counts)
    back_p = back_counts / back_counts.sum()
    print(back_unique)
    print(back_p)

    print(back_deta)

    # 拟合截断幂律分布
    fit = powerlaw.Fit(back_deta, xmin=min(back_deta), discrete=True, estimate_discrete=True, xmin_distance='D')
    print("alpha", fit.power_law.alpha)
    print(f"最优 xmin: {fit.xmin}")
    print(f"最优 xmax: {fit.xmax}")
    print("D:", fit.power_law.D)
    """
    # 初始化最优 xmax 和对应的拟合优度
    best_xmax = None
    best_statistic = float('inf')
    # 遍历不同的 xmax 值
    for xmax in range(100, 300):  # 假设 xmax 的范围是 1000 到 1100
        # 拟合幂律分布
        fit = powerlaw.Fit(back_deta[back_deta <= xmax],xmin=min(back_deta),  discrete=True)

        # 获取拟合的 KS 统计量
        statistic = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)[0]

        # 选择使 KS 统计量最小的 xmax
        if statistic < best_statistic:
            best_statistic = statistic
            best_xmax = xmax

    print(f"Best xmax: {best_xmax}")
    print(f"Best KS Statistic: {best_statistic}")
    """

    # 定义理论的幂律分布 CDF
    def power_law_cdf(x, alpha, xmin):
        return 1 - (xmin / x) ** (alpha - 1)

    # 将原始数据转换为累积分布函数形式
    data_sorted = np.sort(back_deta[back_deta >= fit.xmin])
    data_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # 使用 scipy.stats.kstest 计算 KS 检验的统计量和 p 值
    ks_statistic, p_value = kstest(data_sorted, power_law_cdf, args=(fit.power_law.alpha, fit.xmin))

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    for j, ax in enumerate(axes):

        if j == 0:
            # x = [_ / len(back) for _ in range(len(back))]
            x = [_ / node_count for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_title(network_name, fontsize=10)
            # ax.set_xlim([0, 0.15])
        elif j == 1:
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            fit.power_law.plot_pdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale("linear")
            ax.set_ylabel('P')
            ax.set_yscale("linear")
            ax.legend()
            # ax.set_xlim([0,1000])

            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            fit.power_law.plot_pdf(color='red', ax=ax)
            # 绘制拟合曲线
            # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            # P_fit = truncated_power_law(S_fit, t, S0, b)
            # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
            # fit.power_law.plot_ccdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()


def deta_GSCC_new_mult_1():
    # dir = "pridect_SF_new/"
    dir = "bar_source/"
    file_pre = "SF_1000_2.5_"
    filenames = findfile(dir, file_pre)
    network_names = filenames
    print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)

    num = 30

    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    node_count = g.number_of_nodes()
    back = np.zeros(1000)
    for epoch in range(num):  # 计算平均GSCC曲线
        network_name = network_names[epoch]
        temp = np.load('final_DN_result/' + network_name + '_back.npy')
        temp = temp * node_count
        back += temp / (num)
    # back = np.load('final_DN_result/' + network_name + '_back.npy')
    back_deta = -np.diff(back)
    back_deta = np.round(back_deta, 5)  # 防止多余的小数

    back_deta = back_deta[back_deta != 0]
    # epsilon = 1e-6
    epsilon = 0
    back_deta = back_deta + epsilon
    print(back_deta)
    # 计算变化量的频率分布
    back_unique, back_counts = np.unique(back_deta, return_counts=True)
    print(back_counts)
    back_p = back_counts / back_counts.sum()
    print(back_unique)
    print(back_p)

    print(back_deta)

    # 拟合截断幂律分布
    fit = powerlaw.Fit(back_deta, min=min(back_deta), discrete=True)
    print("alpha", fit.power_law.alpha)

    for j, ax in enumerate(axes):

        if j == 0:
            # x = [_ / len(back) for _ in range(len(back))]
            x = [_ / node_count for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_title(network_name, fontsize=10)
            # ax.set_xlim([0, 0.15])
        elif j == 1:
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            fit.power_law.plot_pdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale("linear")
            ax.set_ylabel('P')
            ax.set_yscale("linear")
            ax.legend()
            # ax.set_xlim([0,1000])

            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            fit.power_law.plot_pdf(color='red', ax=ax)
            # 绘制拟合曲线
            # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            # P_fit = truncated_power_law(S_fit, t, S0, b)
            # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
            # fit.power_law.plot_ccdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()


def deta_GSCC_new_bin_1():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    back = np.load('final_DN_result/' + network_name + '_back_new.npy')
    back = back * node_count
    print(back)
    print(len(back))
    # 从后向前遍历列表，直到遇到第一个大于1的值
    # 从后向前找到第一个大于1的值的位置
    # i = len(back) - 1
    # while i >= 0 and back[i] <= 1:
    #    i -= 1
    # back = back[:i + 1]

    print(back)
    print(len(back))
    back_deta = -np.diff(back)
    back_deta = np.round(back_deta, 5)  # 防止多余的小数

    back_deta = back_deta[back_deta != 0]
    # epsilon = 1e-6
    epsilon = 0
    back_deta = back_deta + epsilon
    # 计算变化量的频率分布
    back_unique, back_counts = np.unique(back_deta, return_counts=True)
    print(back_counts)
    back_p = back_counts / back_counts.sum()
    print(back_unique)
    print(back_p)

    print(back_deta)

    # 将 back_unique 转换为对数坐标
    log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
    # 设置对数坐标上的分箱边界（等距选择 bin）
    # log_min = np.log10(back_unique.min())  # 对数坐标下限
    log_min = 0  # 对数坐标下限
    log_max = np.log10(back_unique.max())  # 对数坐标上限
    bin_edges_log = np.arange(log_min, log_max + 0.2, 0.2)  # 对数等距分箱
    bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
    # 将数据点分配到分箱
    bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
    print(bin_indices)
    # 初始化结果
    binned_unique = []
    binned_p = []
    binned_p1 = []
    binned_p2 = []
    # 遍历每个分箱并计算均值和总和
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            # 分箱内求和
            binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
            log_back_p = np.log10(back_p[bin_mask])
            log_back_mean = log_back_p.mean()
            back_p_mean = np.power(10, log_back_mean)
            binned_p.append(back_p_mean)
            # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
    print(binned_unique)
    print(binned_p)

    """
    # 拟合截断幂律分布
    fit = powerlaw.Fit(back_deta, xmin=min(back_deta), discrete=True, estimate_discrete=True, xmin_distance='D')
    print("alpha", fit.power_law.alpha)
    print(f"最优 xmin: {fit.xmin}")
    print(f"最优 xmax: {fit.xmax}")
    print("D:", fit.power_law.D)
    """
    # 拟合幂律分布
    fit = powerlaw.Fit(binned_unique, xmin=min(binned_unique), discrete=True)
    print("拟合后的 alpha:", fit.power_law.alpha)
    print("拟合后的 xmin:", fit.xmin)
    print("拟合后的 D:", fit.power_law.D)

    """
    # 初始化最优 xmax 和对应的拟合优度
    best_xmax = None
    best_statistic = float('inf')
    # 遍历不同的 xmax 值
    for xmax in range(100, 300):  # 假设 xmax 的范围是 1000 到 1100
        # 拟合幂律分布
        fit = powerlaw.Fit(back_deta[back_deta <= xmax],xmin=min(back_deta),  discrete=True)

        # 获取拟合的 KS 统计量
        statistic = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)[0]

        # 选择使 KS 统计量最小的 xmax
        if statistic < best_statistic:
            best_statistic = statistic
            best_xmax = xmax

    print(f"Best xmax: {best_xmax}")
    print(f"Best KS Statistic: {best_statistic}")
    """

    # 定义理论的幂律分布 CDF
    def power_law_cdf(x, alpha, xmin):
        return 1 - (xmin / x) ** (alpha - 1)

    # 将原始数据转换为累积分布函数形式
    data_sorted = np.sort(back_deta[back_deta >= fit.xmin])
    data_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # 使用 scipy.stats.kstest 计算 KS 检验的统计量和 p 值
    ks_statistic, p_value = kstest(data_sorted, power_law_cdf, args=(fit.power_law.alpha, fit.xmin))

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    for j, ax in enumerate(axes):

        if j == 0:
            # x = [_ / len(back) for _ in range(len(back))]
            x = [_ / node_count for _ in range(len(back))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            ax.set_title("TAD", y=0.8, x=0.5)
            ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            ax.tick_params(labelsize=10)
            ax.set_title(network_name, fontsize=10)
            # ax.set_xlim([0, 0.15])
        elif j == 1:
            # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)

            fit.power_law.plot_pdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale("linear")
            ax.set_ylabel('P')
            ax.set_yscale("linear")
            ax.legend()
            # ax.set_xlim([0,1000])

            # 在图上显示 t 的值
            ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}', transform=ax.transAxes, fontsize=12,
                    ha='center', va='center')
        else:
            # 绘制原始数据
            # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
            fit.power_law.plot_pdf(color='red', ax=ax)
            # 绘制拟合曲线
            # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
            # P_fit = truncated_power_law(S_fit, t, S0, b)
            # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
            # fit.power_law.plot_ccdf(color='red', ax=ax)
            ax.set_title(network_name, fontsize=10)
            ax.set_xlabel('deta_GSCC')
            ax.set_xscale('log')
            ax.set_ylabel('P')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim([0, 1000])
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.show()


def deta_GSCC_new_mult2_1():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axess = plt.subplots(8, 3, figsize=(12, 15))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    method = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]

    def draw_fun(axes, path):
        back = np.load('final_DN_result/' + network_name + "_" + path + '_new.npy')
        back = back * node_count
        print(back)
        print(len(back))
        # 从后向前遍历列表，直到遇到第一个大于1的值
        # 从后向前找到第一个大于1的值的位置
        # i = len(back) - 1
        # while i >= 0 and back[i] <= 1:
        #    i -= 1
        # back = back[:i + 1]

        print(back)
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数

        back_deta = back_deta[back_deta != 0]

        # epsilon = 1e-6
        epsilon = 0
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)
        back_deta_raw = np.repeat(back_unique, np.round(back_p * len(back_deta)).astype(int))
        # 拟合截断幂律分布
        fit = powerlaw.Fit(back_deta_raw, xmin=min(back_deta), discrete=True)
        print("alpha", fit.power_law.alpha)
        print(f"最优 xmin: {fit.xmin}")
        print(f"最优 xmax: {fit.xmax}")
        print("D:", fit.power_law.D)

        # 蒙特卡洛模拟计算 p 值
        D_obs = fit.power_law.D
        D_sim = []
        for _ in range(1000):  # 模拟1000次
            sim_data = fit.power_law.generate_random(len(back_deta))
            sim_fit = powerlaw.Fit(sim_data, xmin=min(back_deta), discrete=True)
            D_sim.append(sim_fit.power_law.D)
            # print("模拟后的 D:", sim_fit.power_law.D)

        p_value = np.sum(np.array(D_sim) >= D_obs) / len(D_sim)
        print(f"KS检验的p值: {p_value:.4f}")

        # 判断KS检验是否通过
        if p_value > 0.05:
            print("KS检验通过，拟合的截断幂律分布与数据一致")
        else:
            print("KS检验未通过，拟合的截断幂律分布与数据不一致")

        for j, ax in enumerate(axes):

            if j == 0:
                # x = [_ / len(back) for _ in range(len(back))]
                x = [_ / node_count for _ in range(len(back))]
                # col.set_title(r'AvgD='+D[epoch],y=0.9)
                # ax.set_title("TAD", y=0.8, x=0.5)
                ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
                ax.tick_params(labelsize=10)
                if path == "back":
                    ax.set_title(network_name, fontsize=10)
                ax.set_ylabel(path)
                # ax.set_xlim([0, 0.15])
            elif j == 1:
                ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale("linear")
                ax.set_ylabel('P')
                ax.set_yscale("linear")
                ax.legend()
                ax.set_ylim(0, 0.65)
                ax.set_xlim(1, 250)

                # 在图上显示 t 的值
                ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}\np = {p_value}', transform=ax.transAxes,
                        fontsize=12,
                        ha='center', va='center')
            else:
                # 绘制原始数据
                ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # 绘制拟合曲线
                # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
                # P_fit = truncated_power_law(S_fit, t, S0, b)
                # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
                # fit.power_law.plot_ccdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale('log')
                ax.set_ylabel('P')
                ax.set_yscale('log')
                ax.legend()
                ax.set_xlim([0, 1000])

    for i, method_name in enumerate(method):
        print(i)
        print(method_name)
        draw_fun(axess[i], method_name)
    plt.suptitle('Difference of deta_GSCC in TAD')
    plt.tight_layout()
    plt.show()


def deta_GSCC_new_mult2_bin_1():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axess = plt.subplots(8, 3, figsize=(12, 15))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    method = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]
    topic_name = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]

    def draw_fun(axes, path, idexi):
        back = np.load('final_DN_result/' + network_name + "_" + path + '_new.npy')
        back = back * node_count
        print(back)
        print(len(back))
        # 从后向前遍历列表，直到遇到第一个大于1的值
        # 从后向前找到第一个大于1的值的位置
        # i = len(back) - 1
        # while i >= 0 and back[i] <= 1:
        #    i -= 1
        # back = back[:i + 1]

        print(back)
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数

        back_deta = back_deta[back_deta != 0]

        # epsilon = 1e-6
        epsilon = 0
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + 0.1, 0.1)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        binned_p1 = []
        binned_p2 = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
        print(binned_unique)
        print(binned_p)

        """
        # 拟合截断幂律分布
        fit = powerlaw.Fit(back_deta, xmin=min(back_deta), discrete=True, estimate_discrete=True, xmin_distance='D')
        print("alpha", fit.power_law.alpha)
        print(f"最优 xmin: {fit.xmin}")
        print(f"最优 xmax: {fit.xmax}")
        print("D:", fit.power_law.D)
        """
        # 拟合幂律分布
        fit = powerlaw.Fit(binned_unique, xmin=min(binned_unique), discrete=True)
        print("拟合后的 alpha:", fit.power_law.alpha)
        print("拟合后的 xmin:", fit.xmin)
        print("拟合后的 D:", fit.power_law.D)

        # 蒙特卡洛模拟计算 p 值
        D_obs = fit.power_law.D
        D_sim = []
        for _ in range(1000):  # 模拟1000次
            sim_data = fit.power_law.generate_random(len(back_deta))
            sim_fit = powerlaw.Fit(sim_data, xmin=min(back_deta), discrete=True, xmin_distance='D')
            D_sim.append(sim_fit.power_law.D)

        p_value = np.sum(np.array(D_sim) >= D_obs) / len(D_sim)
        print(f"KS检验的p值: {p_value:.4f}")

        # 判断KS检验是否通过
        if p_value > 0.05:
            print("KS检验通过，拟合的截断幂律分布与数据一致")
        else:
            print("KS检验未通过，拟合的截断幂律分布与数据不一致")

        for j, ax in enumerate(axes):

            if j == 0:
                # x = [_ / len(back) for _ in range(len(back))]
                x = [_ / node_count for _ in range(len(back))]
                # col.set_title(r'AvgD='+D[epoch],y=0.9)
                # ax.set_title("TAD", y=0.8, x=0.5)
                ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
                ax.tick_params(labelsize=10)
                if path == "back":
                    ax.set_title(network_name, fontsize=10)
                ax.set_ylabel(topic_name[idexi])
                # ax.set_xlim([0, 0.15])
            elif j == 1:
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale("linear")
                ax.set_ylabel('P')
                ax.set_yscale("linear")
                ax.legend()
                ax.set_ylim(0, 0.65)
                ax.set_xlim(1, 250)

                # 在图上显示 t 的值
                # ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}\nD = {fit.power_law.D}', transform=ax.transAxes, fontsize=12, ha='center', va='center')
                ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}\nD={fit.power_law.D}', transform=ax.transAxes,
                        fontsize=12, ha='center', va='center')
            else:
                # 绘制原始数据
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # 绘制拟合曲线
                # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
                # P_fit = truncated_power_law(S_fit, t, S0, b)
                # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
                # fit.power_law.plot_ccdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale('log')
                ax.set_ylabel('P')
                ax.set_yscale('log')
                ax.legend()
                ax.set_xlim([0, 1000])

    for i, method_name in enumerate(method):
        print(i)
        print(method_name)
        draw_fun(axess[i], method_name, i)
    plt.suptitle('Difference of deta_GSCC in different metholds')
    plt.tight_layout()
    plt.show()


def deta_GSCC_new_mult2_bin_Clauset():
    dir = "pridect_SF_new/"
    # dir = "bar_source/"
    # file_pre = "SF_1000_2.5_"
    # filenames = findfile(dir, file_pre)
    # network_names = [filenames[0]]
    # print(network_names)
    # network_names = ['SF_1000_2.8_12.73']
    # network_names = ['SF_50000_2.5_17.48']
    # network_names = ['SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48', 'SF_50000_2.5_17.48']
    # network_names = ["SF_100000_2.5_17.16"]
    network_names = ["SF_10000_2.8_13.3"]
    fig, axess = plt.subplots(8, 3, figsize=(12, 15))
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.07, right=0.97)
    network_name = network_names[0]
    g = nx.read_graphml(dir + network_name)
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边
    # 找到所有强连通分量
    sccs = list(nx.strongly_connected_components(g))
    # 找到最大的强连通分量
    largest_scc = max(sccs, key=len)
    # 创建极大强连通子图
    new_g = g.subgraph(largest_scc).copy()
    # 更新 g 为极大强连通子图
    g = new_g
    node_count = g.number_of_nodes()
    print(node_count)
    method = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]
    topic_name = ["back", "adpDegree", "degree", "MS", "PR", "DND", "FD", "Core"]

    def draw_fun(axes, path, idexi):
        back = np.load('final_DN_result/' + network_name + "_" + path + '_new.npy')
        back = back * node_count
        print(back)
        print(len(back))
        # 从后向前遍历列表，直到遇到第一个大于1的值
        # 从后向前找到第一个大于1的值的位置
        # i = len(back) - 1
        # while i >= 0 and back[i] <= 1:
        #    i -= 1
        # back = back[:i + 1]

        print(back)
        print(len(back))
        back_deta = -np.diff(back)
        back_deta = np.round(back_deta, 5)  # 防止多余的小数

        back_deta = back_deta[back_deta != 0]

        # epsilon = 1e-6
        epsilon = 0
        back_deta = back_deta + epsilon
        # 计算变化量的频率分布
        back_unique, back_counts = np.unique(back_deta, return_counts=True)
        print(back_counts)
        back_p = back_counts / back_counts.sum()
        print(back_unique)
        print(back_p)

        # 将 back_unique 转换为对数坐标
        log_back_unique = np.log10(back_unique + 1e-6)  # +1e-6 避免 log(0)
        # 设置对数坐标上的分箱边界（等距选择 bin）
        # log_min = np.log10(back_unique.min())  # 对数坐标下限
        log_min = 0  # 对数坐标下限
        log_max = np.log10(back_unique.max())  # 对数坐标上限
        bin_edges_log = np.arange(log_min, log_max + 0.05, 0.05)  # 对数等距分箱
        bin_edges = np.power(10, bin_edges_log)  # 转换为普通值
        # 将数据点分配到分箱
        bin_indices = np.digitize(log_back_unique, bin_edges_log)  # 获取每个点所属的分箱索引
        print(bin_indices)
        # 初始化结果
        binned_unique = []
        binned_p = []
        binned_p1 = []
        binned_p2 = []
        # 遍历每个分箱并计算均值和总和
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                # 分箱内求和
                binned_unique.append(back_unique[bin_mask].mean())  # 对数坐标系下的求和
                log_back_p = np.log10(back_p[bin_mask])
                log_back_mean = log_back_p.mean()
                back_p_mean = np.power(10, log_back_mean)
                binned_p.append(back_p_mean)
                # binned_p2.append(back_p[bin_mask].mean())  # 概率总和  for j, ax in enumerate(axes):
        print(binned_unique)
        print(binned_p)

        """
        # 拟合截断幂律分布
        fit = powerlaw.Fit(back_deta, xmin=min(back_deta), discrete=True, estimate_discrete=True, xmin_distance='D')
        print("alpha", fit.power_law.alpha)
        print(f"最优 xmin: {fit.xmin}")
        print(f"最优 xmax: {fit.xmax}")
        print("D:", fit.power_law.D)
        """
        print(back_deta)
        print(len(back_deta))
        binned_deta_raw = np.repeat(binned_unique, np.round(np.array(binned_p) * len(back_deta)).astype(int))
        # 拟合幂律分布
        fit = powerlaw.Fit(binned_deta_raw, xmin=min(binned_unique), discrete=True)
        print("拟合后的 alpha:", fit.power_law.alpha)
        print("拟合后的 xmin:", fit.xmin)
        print("拟合后的 D:", fit.power_law.D)

        # 蒙特卡洛模拟计算 p 值
        D_obs = fit.power_law.D
        D_sim = []
        for _ in range(1000):  # 模拟1000次
            sim_data = fit.power_law.generate_random(len(back_deta))
            sim_fit = powerlaw.Fit(sim_data, xmin=min(back_deta), discrete=True)
            D_sim.append(sim_fit.power_law.D)
            print("模拟后的 D:", sim_fit.power_law.D)

        p_value = np.sum(np.array(D_sim) >= D_obs) / len(D_sim)
        print(f"KS检验的p值: {p_value:.4f}")

        # 判断KS检验是否通过
        if p_value > 0.05:
            print("KS检验通过，拟合的截断幂律分布与数据一致")
        else:
            print("KS检验未通过，拟合的截断幂律分布与数据不一致")

        for j, ax in enumerate(axes):

            if j == 0:
                # x = [_ / len(back) for _ in range(len(back))]
                x = [_ / node_count for _ in range(len(back))]
                # col.set_title(r'AvgD='+D[epoch],y=0.9)
                # ax.set_title("TAD", y=0.8, x=0.5)
                ax.plot(x, back / node_count, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
                ax.tick_params(labelsize=10)
                if path == "back":
                    ax.set_title(network_name, fontsize=10)
                ax.set_ylabel(topic_name[idexi])
                # ax.set_xlim([0, 0.15])
            elif j == 1:
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale("linear")
                ax.set_ylabel('P')
                ax.set_yscale("linear")
                ax.legend()
                ax.set_ylim(0, 0.65)
                ax.set_xlim(1, 250)

                # 在图上显示 t 的值
                # ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}\nD = {fit.power_law.D}', transform=ax.transAxes, fontsize=12, ha='center', va='center')
                ax.text(0.6, 0.4, f'alpha = {fit.power_law.alpha:.4f}\nD={fit.power_law.D}\np={p_value}',
                        transform=ax.transAxes, fontsize=12, ha='center', va='center')
            else:
                # 绘制原始数据
                # ax.scatter(back_unique, back_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                ax.scatter(binned_unique, binned_p, color='b', alpha=0.7, label=f'{network_name}', s=20)
                fit.power_law.plot_pdf(color='red', ax=ax)
                # 绘制拟合曲线
                # S_fit = np.linspace(min(back_unique), max(back_unique), 100)
                # P_fit = truncated_power_law(S_fit, t, S0, b)
                # ax.plot(S_fit, P_fit, color='r', linestyle='--', label='fit curve')
                # fit.power_law.plot_ccdf(color='red', ax=ax)
                # ax.set_xlabel('deta_GSCC')
                ax.set_xscale('log')
                ax.set_ylabel('P')
                ax.set_yscale('log')
                ax.legend()
                ax.set_xlim([0, 1000])

    for i, method_name in enumerate(method):
        print(i)
        print(method_name)
        draw_fun(axess[i], method_name, i)
    plt.suptitle('Difference of deta_GSCC in different metholds')
    plt.tight_layout()
    plt.show()


# draw_sf_er('SF')            #画SF网络每个参数30个图的平均瓦解曲线
# draw_sf_er('ER')            #画ER网络每个参数30个图的平均瓦解曲线

# draw_sf_er_single("SF")     #画SF网络每个参数1个图的瓦解曲线

# draw_F()              #画SF网络瓦解过程中的F值变化曲线

# deta_GSCC()             #画SF网络的每次瓦解一个点后GSCC变化量的频率分布以及采用bin操作后
# deta_GSCC_mult()        #画多个方法的detaGSCC频率分布
# deta_GSCC_mult_CDF()   #多个方法的detaGSCC的CDF曲线
# deta_GSCC_mult_new()       #幂律拟合并计算一些参数
# deta_GSCC_mult_zhibiao()        #画单个网络不同方法的各个指标的对比
# deta_GSCC_mult_zhibiao_mean()   #画不同方法最大雪崩的对比,30个网络取平均   TAD论文中的fig5


# 自定义函数拟合
# deta_GSCC_new()             #幂律拟合单个网络
# deta_GSCC_new_mult()         #拟合30个网络
# deta_GSCC_new_bin()         #函数拟合（bin）
# deta_GSCC_new_mult2()       #8种方法

# powerlaw幂律拟合
# deta_GSCC_new_1()             #拟合单个网络
# deta_GSCC_new_mult_1()         #拟合30个网络
# deta_GSCC_new_bin_1()

# powerlaw幂律拟合并进行Clauset幂律-KS检验
deta_GSCC_new_mult2_1()  # 8种方法
# deta_GSCC_new_mult2_bin_1()
# deta_GSCC_new_mult2_bin_Clauset()