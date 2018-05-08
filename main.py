import numpy as np
from igraph import *
from glob import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from partitioningClasses import *

def main(files):

    data = []
    label = {}
    color = {}
    entries = {}
    x = []
    y = {}
    kmeans = 0.0

    maxk = 7
    method = "nmi"
    np = 6
    tol = 0.2

    color['edgeBetweeness'] = "blue"
    color['fastGreedy'] = "red"
    color['labelPropag'] = "green"
    color['leadingEigen'] = "purple"
    color['multilevel'] = "orange"
    color['walktrap'] = "black"
    color['infoMap'] = "grey"
    color['kmeans'] = "teal"

    label['edgeBetweeness'] = "Edge Betweenness"
    label['fastGreedy'] = "Fast Greedy"
    label['labelPropag'] = "Label Propagation"
    label['leadingEigen'] = "Leading Eigenvector"
    label['multilevel'] = "Multilevel"
    label['walktrap'] = "Walktrap"
    label['infoMap'] = "Infomap"
    label['kmeans'] = "K-Means"

    entries['edgeBetweeness'] = True
    entries['fastGreedy'] = True
    entries['labelPropag'] = True
    entries['leadingEigen'] = True
    entries['multilevel'] = True
    entries['walktrap'] = True
    entries['infoMap'] = True
    entries['kmeans'] = True

    y  = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : []}

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
            ret = solveGraphs(entries, connectedGraphs, communities, metric_method=method)
            kmeans = solveIAMethods(connectedLists, communities, methods=[method]) 
            consensus(entries, connectedGraphs, communities, method, np, tol)

            ret['kmeans'] = kmeans[0]/len(connectedLists)
            return
            for key, value in ret.iteritems():
                y[key].append(value)
            
            x.append(k)      
    
    plt.style.use('fivethirtyeight')

    for key, val in y.iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("%s(k)" % method)
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()

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


files = sorted(glob("./*.arff"))

#EvaluateKMeans(files,["nmi", "rand"])

main(files)
