from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames



def draw_bar_c_new_SF():   #绘制多个网络的bar
    dir = "F_SF_new"
    D = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
         0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
         0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
         0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,
         0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,
         0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,
         0.7, 0.75, 0.8, 0.85, 0.9]
    file_list = []
    for i in range(len(D)):
        file_list.append("SF_1000_2.8_" + str(D[i]) + "_")
    print(file_list)
    print(len(file_list))

    #unit_topics = ["STLD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
    unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD"]
    #color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in range(len(file_list)):
        print("epoch",epoch)
        file_pre = file_list[epoch]
        print(file_pre)
        network_names = findfile(dir, file_pre)
        back_auc = 0
        degree_auc = 0
        finder_auc = 0
        MS_auc = 0
        adpDegree_auc = 0
        PR_auc = 0
        DND_auc = 0
        Core_auc = 0
        #control_auc = 0
        ba = []
        da = []
        fa = []
        ma = []
        aa = []
        pa = []
        dnda = []
        corea = []
        #controla = []
        num = len(network_names)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            print('网络名称：', name)
            back = np.load('final_DN_result/' + name + '_back.npy')
            #print(back)
            # print(len(back))
            degree = np.load('final_DN_result/' + name + '_degree.npy')
            # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
            finder = np.load('final_DN_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
            # print(name)
            #print(finder)

            MS = np.load('final_DN_result/' + name + '_MS.npy')
            adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
            PR = np.load('final_DN_result/' + name + '_PR.npy')
            DND = np.load('final_DN_result/' + name + '_DND.npy')
            CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
            #control = np.load('final_DN_result/' + name + '_control.npy')  # control1才是对的，control是先删除最长路径中的所有点，而不是源头点

            max_val = max(back)

            back_auc += back.sum() / (1000 * max_val)
            ba.append(back.sum() / (1000 * max_val))
            degree_auc += degree.sum() / (1000 * max_val)
            da.append(degree.sum() / (1000 * max_val))
            finder_auc += finder.sum() / (1000 * max_val)
            fa.append(finder.sum() / (1000 * max_val))
            # print(finder_auc)
            # print(fa)
            MS_auc += MS.sum() / (1000 * max_val)
            ma.append(MS.sum() / (1000 * max_val))
            adpDegree_auc += adpDegree.sum() / (1000 * max_val)
            aa.append(adpDegree.sum() / (1000 * max_val))
            PR_auc += PR.sum() / (1000 * max_val)
            pa.append(PR.sum() / (1000 * max_val))
            DND_auc += DND.sum() / (1000 * max_val)
            dnda.append(DND.sum() / (1000 * max_val))
            Core_auc += CoreHD.sum() / (1000 * max_val)
            corea.append(CoreHD.sum() / (1000 * max_val))
            #control_auc += control.sum() / (1000 * max_val)
            #controla.append(control.sum() / (1000 * max_val))
        #std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),np.std(dnda, ddof=1),np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
        std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),
               np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1)]
        #temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,adpDegree_auc / num,degree_auc / num, control_auc / num]
        temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                adpDegree_auc / num, degree_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)
    for topic, data in methods.items():
        # 分别绘制0.8之前的点和0.8之后的点
        if topic == "CoreHD":
            #plt.plot(D[:11], data['mean'][:11], color=color[unit_topics.index(topic)], marker='o', linestyle='-', lw=4, ms=5)
            #plt.plot(D[11:], data['mean'][11:],  color=color[unit_topics.index(topic)], marker='o', linestyle='None')
            #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
            plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
            plt.fill_between(D, np.array(data['mean']) - np.array(data['std']),
                             np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                             alpha=0.2)
            #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
        else:
            #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
            plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
            plt.fill_between(D, np.array(data['mean']) - np.array(data['std']),
                             np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                             alpha=0.2)
            #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
    plt.xlabel('F0', fontsize=12)
    plt.text(0.0, 0.55, 'ANC', va='center', fontsize=12, rotation='vertical')
    plt.legend(bbox_to_anchor=(0.98, 1))
    plt.show()


def draw_bar_c_new_ER():   #绘制多个网络的bar
    dir = "F_ER_new/"
    D = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
         0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
         0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    D2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

    #unit_topics = ["STLD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
    unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD"]
    #color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in D:
        print("epoch",epoch)
        if epoch in D2:
            network_names = findfile(dir, "F_ER_1000_" + str(epoch) + "_")
        else:
            network_names = findfile(dir, "F_1000_" + str(epoch) + "_")
        back_auc = 0
        degree_auc = 0
        finder_auc = 0
        MS_auc = 0
        adpDegree_auc = 0
        PR_auc = 0
        DND_auc = 0
        Core_auc = 0
        #control_auc = 0
        ba = []
        da = []
        fa = []
        ma = []
        aa = []
        pa = []
        dnda = []
        corea = []
        #controla = []
        num = len(network_names)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            print('网络名称：', name)
            back = np.load('final_DN_result/' + name + '_back.npy')
            #print(back)
            # print(len(back))
            degree = np.load('final_DN_result/' + name + '_degree.npy')
            # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
            finder = np.load('final_DN_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
            # print(name)
            #print(finder)

            MS = np.load('final_DN_result/' + name + '_MS.npy')
            adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
            PR = np.load('final_DN_result/' + name + '_PR.npy')
            DND = np.load('final_DN_result/' + name + '_DND.npy')
            CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
            #control = np.load('final_DN_result/' + name + '_control.npy')  # control1才是对的，control是先删除最长路径中的所有点，而不是源头点

            max_val = max(back)

            back_auc += back.sum() / (1000 * max_val)
            ba.append(back.sum() / (1000 * max_val))
            degree_auc += degree.sum() / (1000 * max_val)
            da.append(degree.sum() / (1000 * max_val))
            finder_auc += finder.sum() / (1000 * max_val)
            fa.append(finder.sum() / (1000 * max_val))
            # print(finder_auc)
            # print(fa)
            MS_auc += MS.sum() / (1000 * max_val)
            ma.append(MS.sum() / (1000 * max_val))
            adpDegree_auc += adpDegree.sum() / (1000 * max_val)
            aa.append(adpDegree.sum() / (1000 * max_val))
            PR_auc += PR.sum() / (1000 * max_val)
            pa.append(PR.sum() / (1000 * max_val))
            DND_auc += DND.sum() / (1000 * max_val)
            dnda.append(DND.sum() / (1000 * max_val))
            Core_auc += CoreHD.sum() / (1000 * max_val)
            corea.append(CoreHD.sum() / (1000 * max_val))
            #control_auc += control.sum() / (1000 * max_val)
            #controla.append(control.sum() / (1000 * max_val))
        #std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),np.std(dnda, ddof=1),np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
        std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),
               np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1)]
        #temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,adpDegree_auc / num,degree_auc / num, control_auc / num]
        temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                adpDegree_auc / num, degree_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])
    plt.figure(figsize=(10, 6))
    for topic, data in methods.items():
        # 分别绘制0.8之前的点和0.8之后的点
        if topic == "CoreHD":
            #plt.plot(D[:11], data['mean'][:11], color=color[unit_topics.index(topic)], marker='o', linestyle='-', lw=4, ms=5)
            #plt.plot(D[11:], data['mean'][11:],  color=color[unit_topics.index(topic)], marker='o', linestyle='None')
            #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
            plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
            plt.fill_between(D, np.array(data['mean']) - np.array(data['std']),
                             np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                             alpha=0.2)
            #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
        else:
            #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
            plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
            plt.fill_between(D, np.array(data['mean']) - np.array(data['std']),
                             np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                             alpha=0.2)
            #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
    plt.xlabel('F0', fontsize=12)
    plt.text(0.0, 0.55, 'ANC', va='center', fontsize=12, rotation='vertical')
    plt.legend(bbox_to_anchor=(0.98, 1))
    plt.show()


def draw_bar_c_new_SFandER():   #绘制多个网络的bar
    def find_intersection(x, y1, y2, threshold=0.2):
        """计算两条曲线的交点，并返回大于阈值的交点"""
        f1 = interp1d(x, y1, kind='linear', fill_value="extrapolate")
        f2 = interp1d(x, y2, kind='linear', fill_value="extrapolate")
        diff = f1(x) - f2(x)
        idx = np.where(np.diff(np.sign(diff)))[0]  # 找到符号变化的位置

        for i in idx:
            intersection_x = x[i] + (x[i + 1] - x[i]) * (f2(x[i]) - f1(x[i])) / (
                        f1(x[i + 1]) - f2(x[i + 1]) + f2(x[i]) - f1(x[i]))
            intersection_y = f1(intersection_x)
            if intersection_y > threshold:  # 检查交点的 y 值是否大于阈值
                return intersection_x, intersection_y

        return None

    # 创建一个大图，包含两个子图
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(top=0.96,  left=0.08, bottom=0.15, right=0.98, wspace=0.2)
    dir = "F_SF_new/"
    D = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
         0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
         0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
         0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,
         0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,
         0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,
         0.7, 0.75, 0.8, 0.85, 0.9]
    file_list = []
    for i in range(len(D)):
        file_list.append("SF_1000_2.8_" + str(D[i]) + "_")
    print(file_list)
    print(len(file_list))

    #unit_topics = ["STLD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
    unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD"]
    #color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in range(len(file_list)):
        print("epoch",epoch)
        file_pre = file_list[epoch]
        print(file_pre)
        network_names = findfile(dir, file_pre)
        back_auc = 0
        degree_auc = 0
        finder_auc = 0
        MS_auc = 0
        adpDegree_auc = 0
        PR_auc = 0
        DND_auc = 0
        Core_auc = 0
        #control_auc = 0
        ba = []
        da = []
        fa = []
        ma = []
        aa = []
        pa = []
        dnda = []
        corea = []
        #controla = []
        num = len(network_names)
        print(num)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            #print('网络名称：', name)
            back = np.load('final_DN_result/' + name + '_back.npy')
            #print(back)
            # print(len(back))
            degree = np.load('final_DN_result/' + name + '_degree.npy')
            # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
            finder = np.load('final_DN_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
            # print(name)
            #print(finder)

            MS = np.load('final_DN_result/' + name + '_MS.npy')
            adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
            PR = np.load('final_DN_result/' + name + '_PR.npy')
            DND = np.load('final_DN_result/' + name + '_DND.npy')
            CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
            #control = np.load('final_DN_result/' + name + '_control.npy')  # control1才是对的，control是先删除最长路径中的所有点，而不是源头点


            """back[back < 0.1] = 0
            degree[degree < 0.1] = 0
            finder[finder < 0.1] = 0
            MS[MS < 0.1] = 0
            adpDegree[adpDegree < 0.1] = 0
            PR[PR < 0.1] = 0
            DND[DND < 0.1] = 0
            CoreHD[CoreHD < 0.1] = 0"""


            max_val = max(max(back),max(degree),max(finder),max(MS),max(adpDegree),max(PR),max(DND),max(CoreHD))
            max_val=1
            #print(max_val)

            if max_val >= 0.1:
                back_auc += back.sum() / (1000 * max_val)
                ba.append(back.sum() / (1000 * max_val))
                degree_auc += degree.sum() / (1000 * max_val)
                da.append(degree.sum() / (1000 * max_val))
                finder_auc += finder.sum() / (1000 * max_val)
                fa.append(finder.sum() / (1000 * max_val))
                # print(finder_auc)
                # print(fa)
                MS_auc += MS.sum() / (1000 * max_val)
                ma.append(MS.sum() / (1000 * max_val))
                adpDegree_auc += adpDegree.sum() / (1000 * max_val)
                aa.append(adpDegree.sum() / (1000 * max_val))
                PR_auc += PR.sum() / (1000 * max_val)
                pa.append(PR.sum() / (1000 * max_val))
                DND_auc += DND.sum() / (1000 * max_val)
                dnda.append(DND.sum() / (1000 * max_val))
                Core_auc += CoreHD.sum() / (1000 * max_val)
                corea.append(CoreHD.sum() / (1000 * max_val))
                #control_auc += control.sum() / (1000 * max_val)
                #controla.append(control.sum() / (1000 * max_val))
            else:
                back_auc += 0
                ba.append(0)
                degree_auc += 0
                da.append(0)
                finder_auc += 0
                fa.append(0)
                # print(finder_auc)
                # print(fa)
                MS_auc += 0
                ma.append(0)
                adpDegree_auc += 0
                aa.append(0)
                PR_auc += 0
                pa.append(0)
                DND_auc += 0
                dnda.append(0)
                Core_auc += 0
                corea.append(0)

        #std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),np.std(dnda, ddof=1),np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
        std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),
               np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1)]
        #temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,adpDegree_auc / num,degree_auc / num, control_auc / num]
        temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                adpDegree_auc / num, degree_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])


    # 找到STLD曲线与其他曲线的第一个交点
    first_not_min_index = None
    first_not_min_f = None
    first_not_min_mean = None

    for topic, data in methods.items():
        if topic != "TAD":
            intersection = find_intersection(D, methods["TAD"]["mean"], data["mean"])
            if intersection:
                intersection_x, intersection_y = intersection
                if first_not_min_index is None or intersection_x < first_not_min_f:
                    first_not_min_index = np.argmin(np.abs(np.array(D) - intersection_x))
                    first_not_min_f = intersection_x
                    first_not_min_mean = intersection_y

    if first_not_min_index is not None:
        for topic, data in methods.items():
            # 分别绘制0.8之前的点和0.8之后的点
            if topic == "CoreHD":
                #plt.plot(D[:11], data['mean'][:11], color=color[unit_topics.index(topic)], marker='o', linestyle='-', lw=4, ms=5)
                #plt.plot(D[11:], data['mean'][11:],  color=color[unit_topics.index(topic)], marker='o', linestyle='None')
                #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
                axs[1].plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
                axs[1].fill_between(D, np.array(data['mean']) - np.array(data['std']),
                                 np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                                 alpha=0.2)
                #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
            else:
                #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
                axs[1].plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
                axs[1].fill_between(D, np.array(data['mean']) - np.array(data['std']),
                                 np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                                 alpha=0.2)
                D_extended = np.linspace(0, 1.05, 10000)
                axs[1].fill_between(D_extended, -0.05, 0.55, where=(D_extended >= 0.6), color='#DDB892', alpha=0.05)

                # 绘制短线
                axs[1].plot([first_not_min_f, first_not_min_f + 0.05],
                               [first_not_min_mean, first_not_min_mean - 0.02],
                               color='black', linestyle='-', linewidth=1)
                # 在短线末端标注字母
                axs[1].text(first_not_min_f + 0.12, first_not_min_mean - 0.03, f"$F$={first_not_min_f:.2f}",
                               ha='right', va='top', fontsize=12, color='black')
                #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
    axs[1].set_xlabel(r'$F$', fontsize=12)
    #axs[1].text(-0.05, 0.5, 'ANC', va='center', fontsize=12, rotation='vertical')
    axs[1].set_ylim(-0.05, 0.55)
    axs[1].set_xlim(0.05, 0.95)
    #axs[1].legend(bbox_to_anchor=(0.7, 0.45), frameon=False)



    dir = "F_ER_new/"
    D = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
         0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,
         0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    D2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

    #unit_topics = ["STLD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
    unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD"]
    #color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
    color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in D:
        print("epoch",epoch)
        if epoch in D2:
            network_names_0 = findfile(dir, "F_ER_1000_" + str(epoch) + "_")
            network_names_1 = findfile(dir, "F_1000_" + str(epoch) + "_")
            network_names = network_names_0 + network_names_1
        else:
            network_names = findfile(dir, "F_1000_" + str(epoch) + "_")
        back_auc = 0
        degree_auc = 0
        finder_auc = 0
        MS_auc = 0
        adpDegree_auc = 0
        PR_auc = 0
        DND_auc = 0
        Core_auc = 0
        #control_auc = 0
        ba = []
        da = []
        fa = []
        ma = []
        aa = []
        pa = []
        dnda = []
        corea = []
        #controla = []
        num = len(network_names)
        print(num)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            #print('网络名称：', name)
            back = np.load('final_DN_result/' + name + '_back.npy')
            #print(back)
            # print(len(back))
            degree = np.load('final_DN_result/' + name + '_degree.npy')
            # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
            finder = np.load('final_DN_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
            # print(name)
            #print(finder)

            MS = np.load('final_DN_result/' + name + '_MS.npy')
            adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
            PR = np.load('final_DN_result/' + name + '_PR.npy')
            DND = np.load('final_DN_result/' + name + '_DND.npy')
            CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
            #control = np.load('final_DN_result/' + name + '_control.npy')  # control1才是对的，control是先删除最长路径中的所有点，而不是源头点


            """
            back[back < 0.1] = 0
            degree[degree < 0.1] = 0
            finder[finder < 0.1] = 0
            MS[MS < 0.1] = 0
            adpDegree[adpDegree < 0.1] = 0
            PR[PR < 0.1] = 0
            DND[DND < 0.1] = 0
            CoreHD[CoreHD < 0.1] = 0"""
            #print(back)


            max_val = max(max(back),max(degree),max(finder),max(MS),max(adpDegree),max(PR),max(DND),max(CoreHD))
            max_val = 1
            print(max_val)
            if max_val >= 0.1:
                back_auc += back.sum() / (1000 * max_val)
                ba.append(back.sum() / (1000 * max_val))
                degree_auc += degree.sum() / (1000 * max_val)
                da.append(degree.sum() / (1000 * max_val))
                finder_auc += finder.sum() / (1000 * max_val)
                fa.append(finder.sum() / (1000 * max_val))
                # print(finder_auc)
                # print(fa)
                MS_auc += MS.sum() / (1000 * max_val)
                ma.append(MS.sum() / (1000 * max_val))
                adpDegree_auc += adpDegree.sum() / (1000 * max_val)
                aa.append(adpDegree.sum() / (1000 * max_val))
                PR_auc += PR.sum() / (1000 * max_val)
                pa.append(PR.sum() / (1000 * max_val))
                DND_auc += DND.sum() / (1000 * max_val)
                dnda.append(DND.sum() / (1000 * max_val))
                Core_auc += CoreHD.sum() / (1000 * max_val)
                corea.append(CoreHD.sum() / (1000 * max_val))
                # control_auc += control.sum() / (1000 * max_val)
                # controla.append(control.sum() / (1000 * max_val))
            else:
                back_auc += 0
                ba.append(0)
                degree_auc += 0
                da.append(0)
                finder_auc += 0
                fa.append(0)
                # print(finder_auc)
                # print(fa)
                MS_auc += 0
                ma.append(0)
                adpDegree_auc += 0
                aa.append(0)
                PR_auc += 0
                pa.append(0)
                DND_auc += 0
                dnda.append(0)
                Core_auc += 0
                corea.append(0)
                # control_auc += control.sum() / (1000 * max_val)
                # controla.append(control.sum() / (1000 * max_val))
        print(PR / PR[0])
        #std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),np.std(dnda, ddof=1),np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
        std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1),
               np.std(dnda, ddof=1), np.std(aa, ddof=1), np.std(da, ddof=1)]
        #temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,adpDegree_auc / num,degree_auc / num, control_auc / num]
        temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                adpDegree_auc / num, degree_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])


    # 找到STLD曲线与其他曲线的第一个交点
    first_not_min_index = None
    first_not_min_f = None
    first_not_min_mean = None

    for topic, data in methods.items():
        if topic != "TAD":
            intersection = find_intersection(D, methods["TAD"]["mean"], data["mean"])
            if intersection:
                intersection_x, intersection_y = intersection
                if first_not_min_index is None or intersection_x < first_not_min_f:
                    first_not_min_index = np.argmin(np.abs(np.array(D) - intersection_x))
                    first_not_min_f = intersection_x
                    first_not_min_mean = intersection_y

    if first_not_min_index is not None:
        #first_not_min_f = D[first_not_min_index]
        #first_not_min_mean = stld_mean[first_not_min_index]
        for topic, data in methods.items():
            # 分别绘制0.8之前的点和0.8之后的点
            if topic == "CoreHD":
                #plt.plot(D[:11], data['mean'][:11], color=color[unit_topics.index(topic)], marker='o', linestyle='-', lw=4, ms=5)
                #plt.plot(D[11:], data['mean'][11:],  color=color[unit_topics.index(topic)], marker='o', linestyle='None')
                #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
                axs[0].plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
                axs[0].fill_between(D, np.array(data['mean']) - np.array(data['std']),
                                 np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                                 alpha=0.2)
                #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
            else:
                #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
                axs[0].plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
                axs[0].fill_between(D, np.array(data['mean']) - np.array(data['std']),
                                 np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)],
                                 alpha=0.2)
                # 绘制短线
                axs[0].plot([first_not_min_f, first_not_min_f + 0.05],
                               [first_not_min_mean, first_not_min_mean - 0.02],
                               color='black', linestyle='-', linewidth=1)
                # 在短线末端标注字母
                axs[0].text(first_not_min_f + 0.12, first_not_min_mean - 0.03, f"$F$={first_not_min_f:.2f}",
                               ha='right', va='top', fontsize=12, color='black')
                #plt.plot(D[11:], data['mean'][11:], label=topic, color=color[unit_topics.index(topic)], marker='o',linestyle='None')
    axs[0].set_xlabel(r'$F$', fontsize=12)
    axs[0].set_ylim(-0.05, 0.55)
    axs[0].set_xlim(0.05, 0.95)
    #axs[0].legend(bbox_to_anchor=(0.5, 0.48), frameon=False)
    axs[0].text(-0.08, 0.25, 'AUC', va='center', fontsize=12, rotation='vertical')

    axs[0].tick_params(axis='both', labelsize=12)  # 设置横轴和纵轴刻度标签大小为12
    axs[1].tick_params(axis='both', labelsize=12)
    #axs[0].set_title(r'ER Networks', fontsize=12, y=1.0, x=0.5)
    #axs[1].set_title(r'SF Networks', fontsize=12, y=1.0, x=0.5)
    axs[0].text(-0.16, 1.02, 'A', transform=axs[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axs[1].text(-0.16, 1.02, 'B', transform=axs[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axs[1].text(0.68, 0.15, 'Reversed Edge', transform=axs[1].transAxes, fontsize=12, va='top', ha='left')
    axs[0].legend(bbox_to_anchor=(0.98, 0.56), frameon=False)
    plt.show()



draw_bar_c_new_SF()         #不同F初值的SF网络ANC曲线
draw_bar_c_new_ER()         #不同F初值的ER网络ANC曲线

draw_bar_c_new_SFandER()    #不同F初值的SF网络和ER网络AUC曲线
