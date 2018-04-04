# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:57:41 2017

@author: laercio

This code is protected by copyrights, if you use to some purpose or change the content, please let me know.
"""

import numpy as np
from igraph import *
from glob import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Bellow we see all the functions

#Comparator to a sort
def comparator(key):
    return key[1]


#This method solves the problem with the algorithm k-Means
def solveWithKMeans(list, community, numberClasses):
    solve = KMeans(n_clusters = numberClasses, random_state = 0).fit(list)
    kmeans = compare_communities(solve.labels_,community,method = "rand")
    return kmeans

def createCommunity(fileName):
    
    list = []
    community = []
    
    fp = open(fileName, "r")
    ok = False

    
    #this foor loop, take all the elements in one input file.
    for line in fp:
        if(ok == True):
            # pega as linhas, retira os caracteres desnecessarios e coloca em uma lista
            list.append([float(i) for i in line.strip('\r\n').split(",")]) 
        if("@data" in line):
            ok = True;
    

    #numero de classes
    numberClasses = int(list[-1][-1])+1

    #numero de features
    numberFeatures = int(len(list[0]))-1

    #numero de vertices
    numberVertices = len(list)

    #deixa a classe de cada elemento sendo um valor inteiro
    for i in range(len(list)):
         community.append(int(list[i][numberFeatures]))
    
    kmeans = solveWithKMeans(list, community, numberClasses)

    return [community,kmeans]
    

def createGraph(fileName, k):
    
    #declaracao das variaveis
    fp = open(fileName, "r")
    list = []
    edgesList = []
    
    ok = False
    
    graph = Graph()
    
    #inicio do codigo "main"
    for line in fp:
        if(ok == True):
            # pega as linhas, retira os caracteres desnecessarios e coloca em uma lista
            list.append([float(i) for i in line.strip('\r\n').split(",")]) 
        if("@data" in line):
            ok = True;
    
    #numero de classes
    numberClasses = int(list[-1][-1])+1

    #numero de features
    numberFeatures = int(len(list[0]))-1
    
    #numero de vertices
    numberVertices = len(list)
    
    #deixa a classe de cada elemento sendo um valor inteiro
    for i in range(len(list)):
        list[i][numberFeatures-1] = int(list[i][numberFeatures-1])
   
    #calcula a distancia euclidiana dos pontos e monta uma lista
    for i in range(len(list)):
        edgesList.append([])
        for j in range(len(list)):
            if i == j:
                continue
            d = sum([ (list[i][x]-list[j][x])*(list[i][x]-list[j][x]) for x in range(numberFeatures)])
            edgesList[i].append([j, d])
        edgesList[i] = sorted(edgesList[i],key=comparator)
    
    #adiciona os vertices no grafo
    graph.add_vertices(numberVertices)
    m = 0
    #Usando o algoritmo knn, pegamos os k vertices mais proximos de u para se conectar.
    for i in range(numberVertices):
        for j in range(k):
            u,_ = edgesList[i][j]
            graph.add_edge(i,u)
    
    graph.simplify()
    
    return [graph,list]

def testGraphs(graphs):
    
    connectedGraphs = []
    connectedLists = []
    
    for i in range(len(graphs)):
        
        if (graphs[i][0].components().giant().vcount() == graphs[i][0].vcount() and graphs[i][0].components().giant().ecount() == graphs[i][0].ecount()):
            connectedGraphs.append(graphs[i][0])
            connectedLists.append(graphs[i][1])
            
    return [connectedGraphs,connectedLists]
        
def solve(graphs, list, community, kmeans, metric_method="rand"):
    
    numberClasses = community[-1][-1]+1    
    randIndex = []
    kmeans = 0.0
    
    for i in range(8):
        randIndex.append(0)

    for i in range(len(graphs)):    
        #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
        
        edgeBetweeness = compare_communities(community[i],graphs[i].community_edge_betweenness(clusters=numberClasses).as_clustering(numberClasses).membership,method=metric_method)
        randIndex[0] += edgeBetweeness 
        
        #vai aglomerando as comunidades ate que nao aumenta a modularidade
        fastGreedy = compare_communities(community[i],graphs[i].community_fastgreedy().as_clustering(numberClasses).membership,method=metric_method)
        randIndex[1] += fastGreedy
        
        # usa o metodo de (label propagation method of Raghavan et al)
        labelPropag = compare_communities(community[i],graphs[i].community_label_propagation().membership,method=metric_method)
        randIndex[2] += labelPropag
        
        #Newman's leading eigenvector
        leadingEigen = compare_communities(community[i],graphs[i].community_leading_eigenvector(numberClasses).membership,method=metric_method)
        randIndex[3] += leadingEigen
        
        #baseado no algoritmo de Blondel et al.
        multilevel = compare_communities(community[i],graphs[i].community_multilevel().membership,method=metric_method)
        randIndex[4] += multilevel
        
        #baseado em random walks, usa o metodo de  Latapy & Pons
        walktrap = compare_communities(community[i],graphs[i].community_walktrap().as_clustering(numberClasses).membership,method=metric_method)
        randIndex[5] += walktrap

        #verify how to take the membership
        infoMap = compare_communities(community[i],graphs[i].community_infomap().membership,method=metric_method)
        randIndex[6] += infoMap

        kmeans += solveWithKMeans(list[i], community[i], numberClasses)        
    
    randIndex[7] = kmeans

    for i in range(8):
        randIndex[i]/=len(graphs)*1.0
    

    return randIndex


files = sorted(glob("./*.arff"))

label = []
color = []
data = []
x = []
kmeans = 0.0

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

maxk = 6

for k in range(1,maxk):
    communities = []
    graphs = []    
    kmeans = 0.0
    for fileName in files:
        graphs.append(createGraph(fileName,k))
        result,curKMeans = createCommunity(fileName)
        kmeans += curKMeans
        communities.append(result)
    print "Created Graphs for %d nearest neighbours" % (k)
    connectedGraphs = []
    connectedLists  = []
    connectedGraphs,connectedLists = testGraphs(graphs)
    
    print len(connectedGraphs)
    if(len(connectedGraphs) >= 8):
        randIndex = solve(connectedGraphs,connectedLists,communities,kmeans,metric_method="rand") # use rand for rand index and nmi for nmi clustering evaluation
        data.append(randIndex)
        print randIndex[6]
        x.append(k)      
        print "It's over"


plt.style.use('fivethirtyeight')

if(len(data) > 0):
    for i in range(len(data[0])):
        y = []
        
        for j in range(len(data)):
            y.append(data[j][i])
        print label[i]
        print y
        plt.plot(x,y,label=label[i],color=color[i])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()
