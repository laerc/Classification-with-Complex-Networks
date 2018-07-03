import Orange

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from glob import *
from igraph import *

from aimethods import *
from aievaluations import *
from partitioningClasses import *

from ast import literal_eval
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator

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

        rank[method] = 0.0

        for i in range(len(best_performance[kmax][method])):
            cur_rank = 1.0
            cur_val  = best_performance[kmax][method][i]
            for key, val in best_performance[kmax].items():
                #Check if the performance of a given method is greater than the others
                if cur_val < val[i]:
                    cur_rank += 1.0

            rank[method] += cur_rank

        rank[method] /= len(best_performance[kmax][method])*1.0

    return rank

def main(files, kmeans_eval, em_eval):

    data = []
    best_performance_nmi = {}
    best_performance_rand = {}

    label = {}
    color = {}
    entries = {}
    x = []
    y = {}

    maxk = 17
    method = "nmi"
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
            
            #ret = solveGraphs(entries, connectedGraphs, communities, metric_method=method)
            '''
            ret, before_and_after = solveWithConsensus(connectedGraphs, communities, method, tol, np)
            '''
            
            print (k) 
            ret = solveWithConsensus(connectedGraphs, communities, method, tol, np)[0]

            ret['nmi'].update({'EM' : [em_eval['nmi'] for ii in range(len(connectedGraphs))]})
            ret['rand'].update({'EM' : [em_eval['rand'] for ii in range(len(connectedGraphs))]})

            ret['nmi'].update({'kmeans' : [kmeans_eval['nmi']  for ii in range(len(connectedGraphs))]})
            ret['rand'].update({'kmeans' : [kmeans_eval['rand'] for ii in range(len(connectedGraphs))]})

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

    cd = Orange.evaluation.compute_CD(avranks_nmi, 10) #tested on 10 datasets
    Orange.evaluation.graph_ranks(avranks_nmi, names, cd=cd, width=6, textspace=1.2)
    plt.show()

    cd = Orange.evaluation.compute_CD(avranks_rand, 10) #tested on 10 datasets
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
    plt.tight_layout(rect=[0.0,0.0,1.3,1])
    
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
    plt.tight_layout(rect=[0.0,0.0,1.3,1])
    plt.show()

def solve():
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

def parse_to_plot(file_name):
    fp = open(file_name, "r")

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

solve()
#parse_to_plot("tmp")