import networkx as nx
import os

def findfile(directory, file_prefix):
    fileList = []
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                fileList.append(os.path.join(root, fileName))
                filenames.append(fileName)
    return fileList,filenames

def graphml2txt():
    #dir = "bar_source/"
    #file_pre = "SF"  # 文件以tes_开头
    #dir = "F_networks/"
    #file_pre = "F"  # 文件以tes_开头
    #dir = "biye_real_network"
    #dir = "F_networks_new_SF"
    #dir = "F_networks_new"
    #dir = "F_networks_new_test_100sf"
    #dir = "F_networks_new_test_40sf"
    #dir = "SF_new_lamda_test/new_0.61/"
    #dir = "F_ER_new_new/"
    #dir = "SF_new"
    #dir = "biye_real_network_new_g"
    dir = "biye_new_g/"
    #dir = "pridect_SF_new/"
    #dir = "SF_new_lamda2-3.5/"
    #dir = "SF_ER_FINDER/ER/"
    file_pre = "out"   # 真实网络
    fileList,filenames = findfile(dir, file_pre)
    # print(fileList)
    epoch=0
    for item in fileList:
        print('epoch:',epoch)
        print(item)
        g=nx.read_graphml(item)
        nodes=g.nodes
        keys=[i for i in range(len(nodes))]
        nodes_idx = dict(zip(nodes, keys))
        g.remove_edges_from(nx.selfloop_edges(g))

        #file=open('MinSum_txt/'+filenames[epoch]+'.txt','w')
        #file = open('biye_new_g/' + filenames[epoch] + '.txt', 'w')
        file = open('1/' + filenames[epoch] + '.txt', 'w')
        #file = open('SF_new/' + filenames[epoch] + '.txt', 'w')
        for edge in g.edges:
            file.write('D ')
            file.write(str(nodes_idx[edge[0]])+' ')
            file.write(str(nodes_idx[edge[1]])+'\n')
        file.close()
        epoch+=1

def txt2txt():
    file_name = "ia-crime-moreno.txt"  # 文件以tes_开头
    f = open('real_network/' + file_name, 'r')
    ba = f.readlines()
    file = open('SF_txt/' + file_name, 'w')
    for edges in ba:
        edge = [int(x) for x in edges.split()]
        file.write('D ')
        file.write(str(edge[0]-1) + ' ')
        file.write(str(edge[1]-1) + '\n')
    file.close()

def txttograph():
    dir = "biye_new/"
    file_pre = ""  # 真实网络
    fileList,filenames = findfile(dir, file_pre)
    print(fileList)
    print(filenames)
    epoch = 0
    for item in fileList:
        print(filenames[epoch])
        #G = nx.read_edgelist(item, create_using=nx.DiGraph(), data=False, delimiter=',')
        G = nx.read_edgelist(item, create_using=nx.DiGraph(), data=False)
        #G = nx.read_gml(item)
        #nx.write_graphml(G, "biye_new_g/"+filenames[epoch][:-4], prettyprint=True)
        nx.write_graphml(G, "biye_new_g/" + filenames[epoch], prettyprint=True)
        epoch+=1

#txt2txt()
graphml2txt()
#txttograph()