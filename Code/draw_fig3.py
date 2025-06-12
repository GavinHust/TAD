import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import powerlaw
from scipy.special import kl_div
from matplotlib.gridspec import GridSpec


def findfile(directory, file_prefix):  # 获取directory路径下所有以file_prefix开头的文件
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


def deta_GSCC_mult_zhibiao_mean():
    dir = "../Data/Synthetic/SF/"
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

            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_back.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_degree.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_adpDegree.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_FD.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_MS.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_PR.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_DND.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_Core.npy')
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
            back = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_control.npy')
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
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_back.npy')
    back = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_Core.npy')
    corehd = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_PR.npy')
    pr = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_degree.npy')
    degree = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_adpDegree.npy')
    adpdegree = temp
    # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_FD.npy')  # 自己改为有向网络的FINDER             #怎么维度变成1001*1了
    finder = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_MS.npy')
    learn = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_DND.npy')
    dnd = temp
    temp = np.load('../Data/DNdata/NPY/Synthetic/SF/' + network_name + '_control.npy')
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
    ax5.text(-0.08, 1.05, 'a', transform=ax5.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax1.text(-0.1, 1.1, 'b', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax2.text(-0.1, 1.1, 'c', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.1, 1.1, 'd', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax4.text(-0.1, 1.1, 'e', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    plt.show()




deta_GSCC_mult_zhibiao_mean()   #画不同方法最大雪崩的对比,30个网络取平均   TAD论文中的fig3
