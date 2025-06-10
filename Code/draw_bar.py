from matplotlib import pyplot as plt
import numpy as np
import os

def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames
def bar(network_name):
    if network_name=='ER':
        unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
        dir = "bar_source/"
        # dir = "SF_new/"
        # file_pre = network_name  # 文件以tes_开头
        file_pre = "ER"
        fileNames = findfile(dir, file_pre)
        fileNames=fileNames[30:]+fileNames[:30]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D = ['3','6','9','12']
        color = [ '#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30
        for j, col in enumerate(axes):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            control_auc = 0
            ba = []
            da = []
            fa = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea = []
            controla = []
            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')
                # print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
                finder = np.load('final_DN_result/' + name + '_FD1.npy')  # FD1比FD0好
                # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
                # print(name)
                # print(finder)

                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                control = np.load('final_DN_result/' + name + '_control.npy')

                max_val = max(back)
                max_val = 1
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
                control_auc += control.sum() / (1000 * max_val)
                controla.append(control.sum() / (1000 * max_val))

            std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1),
                   np.std(fa, ddof=1), np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
            temp = [ back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                    adpDegree_auc / num,
                    degree_auc / num, control_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics, fontsize=10, rotation=60)
            font1 = {
                'weight': 'normal',
                'size': 12,
            }
            col.set_title(r'$ER_{\bar{D}=' + D[j] + '}$', font1, y=1.0, x=0.6)
        fig.text(0.01, 0.55, 'AUC', va='center', fontsize=12, rotation='vertical')
        plt.show()


    if network_name=='SF':
        unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
        dir = "bar_source/"
        #dir = "SF_new/"
        #file_pre = network_name  # 文件以tes_开头
        file_pre = "SF_1000"
        fileNames = findfile(dir, file_pre)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D=['2.2','2.5','2.8','3.2']
        color = ['#403990', "#888888",  "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30
        for j, col in enumerate(axes):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            control_auc = 0
            ba = []
            da = []
            fa = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea =  []
            controla = []
            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')
                #print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                #finder_un = np.load('final_DN_result/' + name + '_finder.npy')
                finder = np.load('final_DN_result/' + name + '_FD.npy')   #FD0是目前最好的版本
                #finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
                #print(name)
                #print(finder)

                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                control= np.load('final_DN_result/' + name + '_control.npy')


                max_val = max(back)
                max_val = 1
                back_auc += back.sum() / (1000*max_val)
                ba.append(back.sum() / (1000*max_val))
                degree_auc += degree.sum() / (1000*max_val)
                da.append(degree.sum() / (1000*max_val))
                finder_auc += finder.sum() / (1000*max_val)
                fa.append(finder.sum() / (1000*max_val))
                # print(finder_auc)
                # print(fa)
                MS_auc += MS.sum() / (1000*max_val)
                ma.append(MS.sum() / (1000*max_val))
                adpDegree_auc += adpDegree.sum() / (1000*max_val)
                aa.append(adpDegree.sum() / (1000*max_val))
                PR_auc += PR.sum() / (1000*max_val)
                pa.append(PR.sum() / (1000*max_val))
                DND_auc += DND.sum() / (1000*max_val)
                dnda.append(DND.sum() / (1000*max_val))
                Core_auc += CoreHD.sum() / (1000*max_val)
                corea.append(CoreHD.sum() / (1000*max_val))
                control_auc += control.sum() / (1000*max_val)
                controla.append(control.sum() / (1000*max_val))

            std = [ np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1), np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
            temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num, adpDegree_auc / num,
                    degree_auc / num, control_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics,fontsize=10, rotation=60)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.01, 0.55, 'AUC', va='center', fontsize=12, rotation='vertical')
        plt.show()







def bar_curve(network_name):
    if network_name=='ER':
        unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
        dir = "bar_source/"
        # dir = "SF_new/"
        # file_pre = network_name  # 文件以tes_开头
        file_pre = "ER"
        fileNames = findfile(dir, file_pre)
        fileNames=fileNames[30:]+fileNames[:30]
        network_names = fileNames  # 调整为参数从小到大的顺序
        lamb = [r'$ER_{\langle d \rangle=3}$', r'$ER_{\langle d \rangle=6}$', r'$ER_{\langle d \rangle=9}$', r'$ER_{\langle d \rangle=12}$']

        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)

        D = ['3','6','9','12']
        color = ['#403990', "#888888", "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30
        for j, col in enumerate(axes[0]):

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
                back += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_Core.npy')
                corehd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_PR.npy')
                pr += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_degree.npy')
                degree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
                adpdegree += temp / (num*max(temp))
                # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
                if type == 'SF':
                    temp = np.load('final_DN_result/' + network_name + '_FD.npy')
                elif type == 'ER':
                    temp = np.load('final_DN_result/' + network_name + '_FD1.npy')  # FD1比FD好
                finder += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_MS.npy')
                learn += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_DND.npy')
                dnd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_control.npy')
                control += temp / (num*max(temp))



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
            col.set_ylim(-0.05, 1.05)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=12, ha='center')
        fig.text(0.02, 0.75, 'GSCC', va='center', fontsize=12, rotation='vertical')
        col.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 9},
                   bbox_to_anchor=(0.6, 1.17), loc=1, ncol=9, borderaxespad=0)

        for j, col in enumerate(axes[1]):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            control_auc = 0
            ba = []
            da = []
            fa = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea = []
            controla = []
            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')
                # print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                # finder_un = np.load('final_DN_result/' + name + '_finder.npy')
                finder = np.load('final_DN_result/' + name + '_FD1.npy')  # FD1比FD0好
                # finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
                # print(name)
                # print(finder)

                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                control = np.load('final_DN_result/' + name + '_control.npy')

                back[back < 0.1] = 0
                degree[degree < 0.1] = 0
                finder[finder < 0.1] = 0
                MS[MS < 0.1] = 0
                adpDegree[adpDegree < 0.1] = 0
                PR[PR < 0.1] = 0
                DND[DND < 0.1] = 0
                CoreHD[CoreHD < 0.1] = 0
                control[control < 0.1] = 0

                max_val = max(back)
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
                    control_auc += control.sum() / (1000 * max_val)
                    controla.append(control.sum() / (1000 * max_val))
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
                    control_auc += 0
                    controla.append(0)

            std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1),
                   np.std(fa, ddof=1), np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
            temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num,
                    adpDegree_auc / num,
                    degree_auc / num, control_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics, fontsize=10, rotation=60)
            col.set_ylim(0, 0.45)
            font1 = {
                'weight': 'normal',
                'size': 12,
            }
            col.set_title(r'$ER_{\langle d \rangle=' + D[j] + '}$', font1, y=1.0, x=0.5)
        fig.text(0.02, 0.25, 'ANC', va='center', fontsize=12, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'A', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        axes[1][0].text(-0.27, 1.1, 'B', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        plt.show()


    if network_name=='SF':
        unit_topics = ["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"]
        dir = "bar_source/"
        #dir = "SF_new/"
        #file_pre = network_name  # 文件以tes_开头
        file_pre = "SF_1000"
        fileNames = findfile(dir, file_pre)
        network_names = findfile(dir, file_pre)
        lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.5}$', r'$SF_{\lambda=2.8}$', r'$SF_{\lambda=3.2}$']
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)

        D=['2.2','2.5','2.8','3.2']
        color = ['#403990', "#888888",  "#00FF00", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000"]
        num = 30


        for j, col in enumerate(axes[0]):

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
                back += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_Core.npy')
                corehd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_PR.npy')
                pr += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_degree.npy')
                degree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
                adpdegree += temp / (num*max(temp))
                # temp=np.load('final_DN_result/' + network_name + '_finder.npy')
                if type == 'SF':
                    temp = np.load('final_DN_result/' + network_name + '_FD.npy')
                elif type == 'ER':
                    temp = np.load('final_DN_result/' + network_name + '_FD1.npy')  # FD1比FD好
                finder += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_MS.npy')
                learn += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_DND.npy')
                dnd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_control.npy')
                control += temp / (num*max(temp))


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
            col.set_ylim(-0.05, 1.05)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=12, ha='center')
        fig.text(0.02, 0.75, 'GSCC', va='center', fontsize=12, rotation='vertical')
        col.legend(["TAD", "CoreHD", "PageRk", 'MinSum', 'FINDER', "DND", 'HDA', "HD", "Control"], prop={'size': 9},
                   bbox_to_anchor=(0.6, 1.17), loc=1, ncol=9, borderaxespad=0)

        for j, col in enumerate(axes[1]):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            control_auc = 0
            ba = []
            da = []
            fa = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea =  []
            controla = []
            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')
                #print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                #finder_un = np.load('final_DN_result/' + name + '_finder.npy')
                finder = np.load('final_DN_result/' + name + '_FD.npy')   #FD0是目前最好的版本
                #finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
                #print(name)
                #print(finder)

                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                control= np.load('final_DN_result/' + name + '_control.npy')


                back[back < 0.1] = 0
                degree[degree < 0.1] = 0
                finder[finder < 0.1] = 0
                MS[MS < 0.1] = 0
                adpDegree[adpDegree < 0.1] = 0
                PR[PR < 0.1] = 0
                DND[DND < 0.1] = 0
                CoreHD[CoreHD < 0.1] = 0
                control[control < 0.1] = 0

                max_val = max(back)
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
                    control_auc += control.sum() / (1000 * max_val)
                    controla.append(control.sum() / (1000 * max_val))
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
                    control_auc += 0
                    controla.append(0)

            std = [np.std(ba, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1), np.std(fa, ddof=1), np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(controla, ddof=1)]
            temp = [back_auc / num, Core_auc / num, PR_auc / num, MS_auc / num, finder_auc / num, DND_auc / num, adpDegree_auc / num,
                    degree_auc / num, control_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics,fontsize=10, rotation=60)
            col.set_ylim(0, 0.6)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.02, 0.25, 'ANC', va='center', fontsize=12, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'A', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        axes[1][0].text(-0.27, 1.1, 'B', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')



        plt.show()



bar('SF')   #AUC
bar('ER')   #AUC

bar_curve("ER")         #ANC
bar_curve("SF")         #ANC
