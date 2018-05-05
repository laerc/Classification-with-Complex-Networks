import numpy as np
from igraph import *
from glob import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from partitioningClasses import *

def main(files):

    data = []
    label = []
    color = []
    entries = {}
    x = []
    kmeans = 0.0

    maxk = 9
    method = "rand"

    label.append("Edge Betweenness")
    color.append("blue")
    label.append("Fast Greedy")
    color.append("red")
    label.append("Label Propagation")
    color.append("green")
    label.append("Leading Eigenvector")
    color.append("purple")
    label.append("Multilevel")
    color.append("orange")
    label.append("Walktrap")
    color.append("black")
    label.append("Infomap")
    color.append("gray")
    label.append("K-Means")
    color.append("teal")

    entries['edgeBetweeness'] = False
    entries['fastGreedy'] = False
    entries['labelPropag'] = False
    entries['leadingEigen'] = False
    entries['multilevel'] = False
    entries['walktrap'] = False
    entries['infoMap'] = False
    entries['kmeans'] = True

    print ("Executing these files : %s" % (files))

    for k in range(1,maxk):
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
            #kmeans = solveIAMethods(connectedLists, communities, metric_method=method) 
            data.append(ret)
            x.append(k)      

    print kmeans, files[0]

    '''
    plt.style.use('fivethirtyeight')

    for i in range(len(data[0])):
        y = []
        
        for j in range(len(data)):
            y.append(data[j][i])
        plt.plot(x,y,label=label[i],color=color[i])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()
    '''

def EvaluateKMeans(files, methods):
	numFiles = 0

	kmeans = []
	dataList = []
	communityList = []

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

print ("File name 				NMI 	Rand Index")

EvaluateKMeans(files,["nmi", "rand"])

#main(files)
