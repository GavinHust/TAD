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
        unit_topics = ["TAD",'MinSum','FINDER','HDA',"HD"]
        dir = "bar_source/"

        file_pre = network_name  # 文件以tes_开头
        fileNames = findfile(dir, file_pre)
        fileNames=fileNames[30:]+fileNames[:30]
        fig, axes = plt.subplots(1, 4,  figsize=(16, 4))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D=['3','6','9','12']
        color=['#403990',"#80A6E2","#FBDD85","#F46F43","#CF3D3E"]
        num=30
        for j, col in enumerate(axes):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            ba=[]
            da=[]
            fa=[]
            ma=[]
            aa=[]
            for i in range(j*num,(j+1)*num):
                name = fileNames[i]#[:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')   #读取瓦解曲线的结果
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                finder = np.load('final_DN_result/' +name + '_finder.npy')
                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                max_val=max(back)

                back_auc += back.sum()/(1000*max_val)   #计算平均auc值
                ba.append(back.sum()/(1000*max_val))
                degree_auc += degree.sum()/(1000*max_val)
                da.append(degree.sum()/(1000*max_val))
                finder_auc += finder.sum()/(1000*max_val)
                fa.append(finder.sum()/(1000*max_val))
                MS_auc += MS.sum()/(1000*max_val)
                ma.append(MS.sum()/(1000*max_val))
                adpDegree_auc += adpDegree.sum()/(1000*max_val)
                aa.append(adpDegree.sum()/(1000*max_val))

            std=[np.std(ba, ddof=1),np.std(ma, ddof=1),np.std(fa, ddof=1),np.std(aa, ddof=1),np.std(da, ddof=1)]
            temp=[back_auc/num,MS_auc/num,finder_auc/num,adpDegree_auc/num,degree_auc/num]
            col.bar(unit_topics, temp, yerr = std,error_kw = {'elinewidth':2,'ecolor' : '0.0', 'capsize' :4 },color=color,width=0.75)
            print(std,temp)
            col.set_xticklabels(unit_topics, Rotation=40)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$ER_{\bar{D}=' + D[j]+'}$', font1,y=1.0,x=0.6)
        fig.text(0.01, 0.55, 'ANC',va='center', fontsize=12, rotation='vertical')
        plt.savefig("final_result/ER_bar.pdf")
        plt.show()
    if network_name=='SF':
        unit_topics = ['TAD', 'TAD_A', "TADj-i", "TADj-i-1", "TADji2"]
        dir = "bar_source/"

        #file_pre = network_name  # 文件以tes_开头
        file_pre = "SF_1000"
        fileNames = findfile(dir, file_pre)
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D=['2.2','2.5','2.8','3.2']
        color = ['#403990',"#80A6E2","#FBDD85","#F46F43","#CF3D3E"]
        num = 30
        for j, col in enumerate(axes):
            back_auc = 0
            backA_auc = 0
            backji_auc = 0
            backji1_auc = 0
            backji2_auc = 0
            degree_auc = 0
            finder_auc = 0
            finderun_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            control_auc = 0
            ba = []
            bAa = []
            bjia = []
            bji1a = []
            bji2a = []
            da = []
            fa = []
            faun = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea =  []
            controla = []
            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')
                backA = np.load('final_DN_result/' + name + '_backA.npy')
                backji = np.load('final_DN_result/' + name + '_backji.npy')
                backji1 = np.load('final_DN_result/' + name + '_backji1.npy')
                backji2 = np.load('final_DN_result/' + name + '_backji2.npy')
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                finder_un = np.load('final_DN_result/' + name + '_finder.npy')
                finder = np.load('final_DN_result/' + name + '_FD.npy')   #FD0是目前最好的版本
                #finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
                #print(name)
                #print(finder)

                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND1.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                control= np.load('final_DN_result/' + name + '_control.npy')


                max_val = max(back)

                back_auc += back.sum() / (1000 * max_val)
                ba.append(back.sum() / (1000 * max_val))

                backA_auc += backA.sum() / (1000 * max_val)
                bAa.append(backA.sum() / (1000 * max_val))

                backji_auc += backji.sum() / (1000 * max_val)
                bjia.append(backji.sum() / (1000 * max_val))

                backji1_auc += backji1.sum() / (1000 * max_val)
                bji1a.append(backji1.sum() / (1000 * max_val))

                backji2_auc += backji2.sum() / (1000 * max_val)
                bji2a.append(backji2.sum() / (1000 * max_val))

                degree_auc += degree.sum() / (1000 * max_val)
                da.append(degree.sum() / (1000 * max_val))
                finder_auc += finder.sum() / (1000 * max_val)
                fa.append(finder.sum() / (1000 * max_val))
                finderun_auc += finder_un.sum() / (1000 * max_val)
                faun.append(finder_un.sum() / (1000 * max_val))
                #print(finder_auc)
                #print(fa)
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

            std = [np.std(ba, ddof=1), np.std(bAa, ddof=1), np.std(bjia, ddof=1), np.std(bji1a, ddof=1), np.std(bji2a, ddof=1)]
            temp = [back_auc / num, backA_auc / num, backji_auc / num, backji1_auc / num, backji2_auc / num]
            bars = col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            """
            # 在每个柱子上方标注数值
            for bar, value in zip(bars, temp):
                height = bar.get_height()
                col.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.5f}', ha='center', va='bottom')
            """
            print(std, temp)
            col.set_xticklabels(unit_topics, rotation=40)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.01, 0.55, 'ANC', va='center', fontsize=12, rotation='vertical')
        plt.show()

bar('SF')   #可选'ER'或'SF'