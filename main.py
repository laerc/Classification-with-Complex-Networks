import Orange

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from glob import *
from igraph import *
from partitioningClasses import *
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator

def EvaluateKMeans(files):
    numFiles = 0

    kmeans = {'nmi' : 0, 'rand' : 0}
    dataList = []
    communityList = []
    numberClasses = 0

    for file in files:
        numFiles += 1
        y,x = parseData(file)

        #remove the last colugnm
        for i in range(len(y)):
            numberClasses = int(x[i][-1])+1
            x[i].pop()
        
        dataList.append(x)
        communityList.append(y)
        # This means that we got all the files with the same parameters, but different values
        if(numFiles == 10):
            for i in range(len(dataList)):
                kmeans['nmi']  += solveWithKMeans(dataList[i],np.asarray(communityList[i]),numberClasses,"nmi")
                kmeans['rand'] += solveWithKMeans(dataList[i],np.asarray(communityList[i]),numberClasses,"rand")
            kmeans['nmi'] /=10.0
            kmeans['rand']/=10.0
            return (kmeans)

def EvaluateEM(files):
    numFiles = 0

    dataList = []
    communityList = []
    numberClasses = 0

    for file in files:
        numFiles += 1
        y,x = parseData(file)

        #remove the last element(that indicates the class label)
        for i in range(len(y)):
            numberClasses = int(x[i][-1]+1)
            x[i].pop()
        
        dataList.append(x)
        communityList.append(y)

        if(numFiles == 10):
            EM = {'nmi' : [], 'rand' : []}
            em_nmi_v  = []
            em_rand_v = []
            
            for j in range(len(dataList)):
                EM['nmi'].append(0)
                EM['rand'].append(0)
                
                for i in range(20):
                    em_nmi  = solveWithEM(dataList[j],communityList[j],numberClasses,'nmi')
                    em_rand = solveWithEM(dataList[j],communityList[j],numberClasses,'rand')
                    EM['nmi'][j]  += em_nmi
                    EM['rand'][j] += em_rand

                em_nmi_v.append(EM['nmi'][j]/20.0)
                em_rand_v.append(EM['rand'][j]/20.0)

            numFiles = 0
            communityList = []
            numFiles = 0
            return { 'nmi' : sum(em_nmi_v)/len(dataList), 'nmi_dp'  : np.std(em_nmi_v, dtype=np.float64),
                     'rand': sum(em_rand_v)/len(dataList), 'rand_dp': np.std(em_rand_v, dtype=np.float64) }
            
            dataList = []

def findBestRank(best_performance):
    rank = {}
    methods = [ 'edgeBetweeness', 'fastGreedy', 'labelPropag', 'leadingEigen','multilevel', 
                'walktrap', 'infoMap', 'EM', 'kmeans']

    for method in methods:
        kmax = 0
        max_val = -1.0
        
        for key, val in best_performance.items():
            if(sum(val[method]) > max_val):
                kmax = key
                max_val = sum(val[method])
        #print (method, max_val, best_performance[kmax][method][0])
        rank[method] = 0.0

        for i in range(len(best_performance[kmax][method])):
            cur_rank = 1.0
            cur_val  = best_performance[kmax][method][i]
            for key, val in best_performance[kmax].items():
                #Check if the performance of a given method is greater than the others
                if cur_val < val[i]:
                    cur_rank += 1.0

            rank[method] += cur_rank
        #print method
        rank[method] /= len(best_performance[kmax][method])*1.0
        #print(method, rank[method])

    return rank

def plotRank(ranks, method):
    
    color = {}
    label = {}

    color['edgeBetweeness'] = "blue"
    color['fastGreedy'] = "red"
    color['labelPropag'] = "green"
    color['leadingEigen'] = "purple"
    color['multilevel'] = "orange"
    color['walktrap'] = "black"
    color['infoMap'] = "grey"
    color['EM'] = 'brown'

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    label['EM'] = "Expectation Maximization"
    
    plt.style.use('fivethirtyeight')
    #print ("Debug message : ")
    #print (ranks)

    #for key, value in ranks.items():
    colors = [value for key, value in color.items()]
    plt.scatter([value for key, value in ranks.items()], [0 for i in range(len(ranks))], c=colors,marker="|", s=5000, linewidths=5, alpha=0.9)    
    #label
    plt.xlabel("Rank Medio(%s)" % (method))
    #x-axis values
    plt.xticks([value for key,value in ranks.items()])
    
    #hide y-axis
    plt.yticks([0 for i in range(len(ranks))]," ")
    
    plt.tight_layout(rect=[0,0,0.75,1])
    #set legend 
    patch = [mpatches.Patch(color='blue',   label="Edge Betweenness"),  mpatches.Patch(color='red', label="Fast Greedy"),
             mpatches.Patch(color='green',  label="Label Propagation"), mpatches.Patch(color='purple', label="Leading Eigenvector"),
             mpatches.Patch(color='orange', label="Multilevel"),        mpatches.Patch(color='black', label="Walktrap"),
             mpatches.Patch(color='grey',   label="Infomap"),           mpatches.Patch(color='brown',   label="EM")] 
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=patch)

    plt.show()

def main(files, kmeans_eval, em_eval):

    data = []
    best_performance_nmi = {}
    best_performance_rand = {}
    bests_perform = {"nmi" : {}, "rand" : {}}
    bests_perform["nmi"]  = {"before" : [], "after" : []}
    bests_perform["rand"] = {"before" : [], "after" : []}

    label = {}
    color = {}
    entries = {}
    x = []
    y = {}

    maxk = 17
    method = "rand"
    np = 10
    tol = 0.70
    maxi_eval_nmi = 0.0
    maxi_eval_rand = 0.0

    color['edgeBetweeness'] = "blue"
    color['fastGreedy'] = "red"
    color['labelPropag'] = "green"
    color['leadingEigen'] = "purple"
    color['multilevel'] = "orange"
    color['walktrap'] = "black"
    color['infoMap'] = "grey"
    color['kmeans'] = "teal"
    color['EM'] = "indigo"

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    label['kmeans'] = "K-Means"
    label['EM'] = "Expectation-Maximization"

    entries['edgeBetweeness'] = False
    entries['fastGreedy'] = True
    entries['labelPropag'] = False
    entries['leadingEigen'] = False
    entries['multilevel'] = True
    entries['walktrap'] = True
    entries['infoMap'] = False
    entries['kmeans'] = True
    entries['EM'] = True
    y = {}
    
    y['nmi'] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : [], 'EM' : []}#, 'consensus' : []}

    y['rand'] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : [], 'EM' : []}#, 'consensus' : []}

    for k in range(2,maxk):
        communities = []
        graphs = []    
        kmeans = 0.0
        
        try:
            for fileName in files:
                graphs.append(createGraph(fileName,k))
                result,_ = parseData(fileName)
                communities.append(result)

            connectedGraphs = []
            connectedLists  = []
            connectedGraphs,connectedLists = testGraphs(graphs)
        except:
            break

        if(len(connectedGraphs) >= 8):
            # use rand for rand index and nmi for nmi clustering evaluation
            cur_val_nmi = 0.0
            cur_val_rand = 0.0
            
            #ai_methods = solveIAMethods(connectedLists, communities, methods=['nmi', 'rand'])
            #ret = solveGraphs(entries, connectedGraphs, communities, metric_method=method)
            '''
            for key_method, value_method in ai_methods.items():
                for key_algo, value_algo in value_method.items():
                    y[key_method][key_algo].append(value_algo/len(connectedLists))
            '''
            
            '''
            ret, before_and_after = solveWithConsensus(connectedGraphs, communities, method, tol, np)
            
            bests_perform["nmi"]["before"].append(sum(before_and_after["nmi"]["before"])/len(connectedGraphs)*1.0)
            bests_perform["rand"]["before"].append(sum(before_and_after["rand"]["before"])/len(connectedGraphs)*1.0)
            bests_perform["nmi"]["after"].append(sum(before_and_after["nmi"]["after"])/len(connectedGraphs)*1.0)
            bests_perform["rand"]["after"].append(sum(before_and_after["rand"]["after"])/len(connectedGraphs)*1.0)
            

            ret['nmi'].update ({'EM' : [em_eval['nmi'] for ii in range(len(connectedGraphs))]})
            ret['rand'].update({'EM' : [em_eval['rand'] for ii in range(len(connectedGraphs))]})

            ret['nmi']['kmeans']  = [kmeans_eval['nmi']  for ii in range(len(connectedGraphs))]
            ret['rand']['kmeans'] = [kmeans_eval['rand'] for ii in range(len(connectedGraphs))]
            '''
            #print ("\n--------------------------------------------------------\n")
            
            print (k) 
            ret = solveWithConsensus(connectedGraphs, communities, method, tol, np)[0]

            ret['nmi'].update ({'EM' : [em_eval['nmi'] for ii in range(len(connectedGraphs))]})
            ret['rand'].update({'EM' : [em_eval['rand'] for ii in range(len(connectedGraphs))]})

            ret['nmi']['kmeans']  = [kmeans_eval['nmi']  for ii in range(len(connectedGraphs))]
            ret['rand']['kmeans'] = [kmeans_eval['rand'] for ii in range(len(connectedGraphs))]

            #print ("\n--------------------------------------------------------\n")
            

            best_performance_nmi.update({k : ret['nmi']})
            best_performance_rand.update({k : ret['rand']})
            
            for key, value in ret["nmi"].items():
                if(key != 'kmeans' and key != 'EM'):
                    y['nmi'][key].append(sum(value)/len(connectedGraphs)*1.0)
                    cur_val_nmi = max(cur_val_nmi, sum(value)/len(connectedGraphs)*1.0)
                else:
                    if(key == 'EM'):
                        y['nmi'][key].append(em_eval['nmi'])
                    else:
                        y['nmi'][key].append(kmeans_eval['nmi'])

            for key, value in ret['rand'].items():
                if(key != 'kmeans' and key != 'EM'):
                    y['rand'][key].append(sum(value)/len(connectedGraphs)*1.0)
                    cur_val_rand = max(cur_val_rand, sum(value)/len(connectedGraphs)*1.0)
                else:
                    if(key == 'EM'):
                        y['rand'][key].append(em_eval['rand'])
                    else:
                        y['rand'][key].append(kmeans_eval['rand'])
            
            x.append(k)

            if(cur_val_nmi < maxi_eval_nmi and cur_val_rand < maxi_eval_rand and k >= 10):
                break
            maxi_eval_nmi = max(maxi_eval_nmi,cur_val_nmi)
            maxi_eval_rand = max(maxi_eval_rand,cur_val_rand)

    ranks = findBestRank(best_performance_nmi)

    print ("nmi")
    print (ranks)
    ranks = findBestRank(best_performance_rand)
    print ("rand")
    print (ranks)
    print (x)
    print (y)
    print (bests_perform)

def plot(rank_nmi, rank_rand, x, y):

    color = {}
    label = {}
    names = []
    avranks_nmi = []
    avranks_rand = []
    
    color['edgeBetweeness'] = "blue"
    color['fastGreedy'] = "red"
    color['labelPropag'] = "green"
    color['leadingEigen'] = "purple"
    color['multilevel'] = "orange"
    color['walktrap'] = "black"
    color['infoMap'] = "grey"
    color['kmeans'] = "teal"
    color['EM'] = "indigo"

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    label['kmeans'] = "K-Means"
    label['EM'] = "Expectation-Maximization"

    for key,val in label.items():
        names.append(val)
        avranks_nmi.append(rank_nmi[key])
        avranks_rand.append(rank_rand[key])

    cd = Orange.evaluation.compute_CD(avranks_nmi, 10) #tested on 30 datasets
    Orange.evaluation.graph_ranks(avranks_nmi, names, cd=cd, width=6, textspace=1.2)
    plt.show()

    cd = Orange.evaluation.compute_CD(avranks_rand, 10) #tested on 30 datasets
    Orange.evaluation.graph_ranks(avranks_rand, names, cd=cd, width=6, textspace=1.2)
    plt.show()
    
    plt.style.use('fivethirtyeight')

    for key, val in y['nmi'].items():
        if(key not in label):
            continue
        if(key == 'EM' or key == 'kmeans'):
            plt.plot(x,val,label=label[key],color=color[key], dashes=[5, 3])
        else:
            plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("NMI(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0.00,0.00,1.3,1])
    
    plt.show()

    for key, val in y['rand'].items():
        if(key == 'EM' or key == 'kmeans'):
            plt.plot(x,val,label=label[key],color=color[key], dashes=[5, 3])
        else:
            plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0.00,0.00,1.3,1])
    plt.show()

files = sorted(glob("./*.arff"))

for i in range(0,len(files),10):
    cur_files = []
    
    for j in range(i,i+10):
        cur_files.append(files[j])
    
    print (cur_files[0])
    em_eval     = EvaluateEM(cur_files)
    kmeans_eval = EvaluateKMeans(cur_files)
    main(cur_files, kmeans_eval, em_eval)
    print ("---------------------------------------------------------------------------------------------------")


'''
from glob import *
from ast import literal_eval

fp = open("tmp_file", "r")

ok = False
list = []

for line in fp:
    if line.startswith("---------"):
        file_name = (list[0])
        rank_nmi  = literal_eval(list[2])
        rank_rand = literal_eval(list[4])
        x = literal_eval(list[5])
        y = literal_eval(list[6])

        print (file_name)
        print (rank_nmi)
        print (rank_rand)
        print (x)
        print (y)
        plot(rank_nmi, rank_rand, x, y)
        list = []
    else:
        line = line.replace('\n', '')
        list.append(line)
'''