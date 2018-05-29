import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from glob import *
from igraph import *
from partitioningClasses import *
from sklearn.cluster import KMeans


def EvaluateKMeans(files, methods):
    numFiles = 0

    kmeans = []
    dataList = []
    communityList = []

    print ("File name              NMI     Rand Index")

    for i in range(len(methods)):
        kmeans.append(0.0)

    for file in files:
        
        numFiles += 1
        y,x = parseData(file)

        #remove the last colugnm
        for i in range(len(y)):
            x[i].pop()
        
        dataList.append(x)
        communityList.append(y)
        # This means that we got all the files with the same parameters, but different values
        if(numFiles == 10):

            kmeans = solveIAMethods(dataList,communityList,methods)

            print ("%s %f %f" % (file, kmeans[0]/len(dataList), kmeans[1]/len(dataList))) 
            numFiles = 10
            dataList = []
            communityList = []
            numFiles = 0

def plotPoints():
    list = []

    list.append([-0.319873,0.749928,0])
    list.append([-0.736998,0.962373,0])
    list.append([-0.313376,1.000112,0])
    list.append([-0.241146,0.891822,0])
    list.append([-0.266910,0.594109,0])
    list.append([0.647737,-0.736721,1])
    list.append([0.707669,-0.618839,1])
    list.append([0.698880,-0.662611,1])
    list.append([0.363784,-0.862940,1])
    list.append([0.550517,-0.792845,1])
    list.append([0.544720,0.673590,2])
    list.append([0.639593,0.978922,2])
    list.append([0.576532,0.733509,2])
    list.append([0.723469,0.758321,2])
    list.append([0.508128,1.153752,2])
    list.append([0.238664,0.814264,3])
    list.append([0.472276,0.493981,3])
    list.append([0.302689,0.366012,3])
    list.append([0.290772,0.883492,3])
    list.append([0.577553,0.646007,3])
    list.append([0.745545,0.269755,4])
    list.append([0.630065,-0.024022,4])
    list.append([0.631174,0.319657,4])
    list.append([0.884768,0.392257,4])
    list.append([0.812949,0.187561,4])
    list.append([0.438302,0.816265,5])
    list.append([0.388258,0.937304,5])
    list.append([0.383612,1.220812,5])
    list.append([0.299978,0.992304,5])
    list.append([0.486339,0.760292,5])
    list.append([0.642726,-0.082337,6])
    list.append([0.603145,-0.509367,6])
    list.append([0.692439,-0.651165,6])
    list.append([0.623191,-0.495239,6])
    list.append([0.613643,-0.258398,6])
    list.append([-0.618423,0.108406,7])
    list.append([-0.424482,0.067517,7])
    list.append([-0.698008,0.075484,7])
    list.append([-0.602790,0.044434,7])
    list.append([-0.563423,-0.003829,7])
    list.append([-0.014894,-0.008668,8])
    list.append([0.051090,-0.248600,8])
    list.append([0.247227,-0.273197,8])
    list.append([0.393160,-0.471588,8])
    list.append([0.082455,-0.250478,8])
    list.append([-0.018573,0.356482,9])
    list.append([0.069887,0.239055,9])
    list.append([0.127577,0.138147,9])
    list.append([0.316402,0.324715,9])
    list.append([0.100229,0.076685,9])

    colors = ['black','blue','red','yellow','purple', 'gray','purple','pink','orange','teal']

    plt.style.use('fivethirtyeight')

    for x in list:
        plt.scatter(x[0], x[1],c=colors[x[2]],marker="o", s=250, linewidths=1, alpha=0.8)
    plt.show()

def findBestRank(best_performance):
    rank = {}
    methods = [ 'edgeBetweeness', 'fastGreedy', 'labelPropag', 'leadingEigen','multilevel', 
                'walktrap', 'infoMap']

    for method in methods:
        kmax = 0
        max_val = 0.0
        
        for key, val in best_performance.iteritems():
            if(sum(val[method]) > max_val):
                kmax = key
                max_val = sum(val[method])

        rank[method] = 0.0

        for i in range(len(best_performance[kmax][method])):
            cur_rank = 1.0
            cur_val  = best_performance[kmax][method][i]
            for key, val in best_performance[kmax].iteritems():
                #Check if the performance of a given method is greater than the others
                if cur_val < val[i]:
                    cur_rank += 1.0

            rank[method] += cur_rank
        print method
        rank[method] /= len(best_performance[kmax][method])*1.0

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

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    
    plt.style.use('fivethirtyeight')
    print "Debug message : "
    print ranks

    #for key, value in ranks.iteritems():
    colors = [value for key, value in color.iteritems()]
    plt.scatter([value for key, value in ranks.iteritems()], [0 for i in range(len(ranks))], c=colors,marker="|", s=5000, linewidths=5, alpha=0.9)    
    #label
    plt.xlabel("Rank Medio(%s)" % (method))
    #x-axis values
    plt.xticks([value for key,value in ranks.iteritems()])
    
    #hide y-axis
    plt.yticks([0 for i in range(len(ranks))]," ")
    
    plt.tight_layout(rect=[0,0,0.75,1])
    #set legend 
    patch = [mpatches.Patch(color='blue',   label="Edge Betweenness"),  mpatches.Patch(color='red', label="Fast Greedy"),
             mpatches.Patch(color='green',  label="Label Propagation"), mpatches.Patch(color='purple', label="Leading Eigenvector"),
             mpatches.Patch(color='orange', label="Multilevel"),        mpatches.Patch(color='black', label="Walktrap"),
             mpatches.Patch(color='grey',   label="Infomap")] 
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=patch)

    plt.show()

def main(files):

    data = []
    best_performance = {}
    label = {}
    color = {}
    entries = {}
    x = []
    y = {}
    kmeans = 0.0

    maxk = 7
    method = "nmi"
    np = 6
    tol = 0.70

    color['edgeBetweeness'] = "blue"
    color['fastGreedy'] = "red"
    color['labelPropag'] = "green"
    color['leadingEigen'] = "purple"
    color['multilevel'] = "orange"
    color['walktrap'] = "black"
    color['infoMap'] = "grey"
    color['kmeans'] = "teal"
    color['consensus'] = "indigo"

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    label['kmeans'] = "K-Means"
    label['consensus'] = "Consensus"

    entries['edgeBetweeness'] = True
    entries['fastGreedy'] = True
    entries['labelPropag'] = True
    entries['leadingEigen'] = True
    entries['multilevel'] = True
    entries['walktrap'] = True
    entries['infoMap'] = True
    entries['kmeans'] = True

    y  = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : []}#, 'consensus' : []}

    print ("Executing these files : %s" % (files))

    for k in range(2,maxk):
        communities = []
        graphs = []    
        kmeans = 0.0
        for fileName in files:
            graphs.append(createGraph(fileName,k))
            result,_ = parseData(fileName)
            communities.append(result)

        connectedGraphs = []
        connectedLists  = []
        connectedGraphs,connectedLists = testGraphs(graphs)

        if(len(connectedGraphs) >= 8):
            # use rand for rand index and nmi for nmi clustering evaluation
            print (k)
            ret = solveGraphs(entries, connectedGraphs, communities, metric_method=method)
            kmeans = solveIAMethods(connectedLists, communities, methods=[method]) 
            y['kmeans'].append(kmeans[0]/len(connectedLists))

            ret_consensus = solveWithConsensus(connectedGraphs, communities, method, tol, np)
            #y['consensus'].append(ret_consensus)
            
            best_performance.update({k : ret})
            for key, value in ret.iteritems():
                y[key].append(sum(value)/len(connectedGraphs)*1.0)

            x.append(k)      

    ranks = findBestRank(best_performance)
    plotRank(ranks, method)

    plt.style.use('fivethirtyeight')

    for key, val in y.iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("%s(k)" % method)
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()


files = sorted(glob("./*.arff"))

#EvaluateKMeans(files,["nmi", "rand"])

main(files)

