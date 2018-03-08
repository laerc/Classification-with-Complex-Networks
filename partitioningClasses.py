# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:57:41 2017

@author: laercio
"""

import numpy as np
from igraph import *
from glob import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#funcoes
#desvio padrao usando a formual : 
def desvioPadrao(data):
    u = np.mean(data)
    return np.sqrt(sum([(x - u)*(x - u) for x in data])/len(data))
    
def comparator(key):
    return key[1]

def solveWithKMeans(list, community, numberClasses):
    solve = KMeans(n_clusters = numberClasses, random_state = 0).fit(list)
    kmeans = compare_communities(solve.labels_,community,method="rand")
    return kmeans

def createCommunity(fileName):
    
    ok = False
    
    list = []
    community = []
    
    fp = open(fileName, "r")
    
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
        
def solve(graphs, list, community, kmeans):
    
    numberClasses = community[-1][-1]+1    
    randIndex = []
    kmeans = 0.0
    
    for i in range(7):
        randIndex.append(0)

    for i in range(len(graphs)):    
        #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
        
        edgeBetweeness = compare_communities(community[i],graphs[i].community_edge_betweenness(clusters=numberClasses).as_clustering(numberClasses).membership,method="rand")
        randIndex[0] += edgeBetweeness 
        
        #vai aglomerando as comunidades ate que nao aumenta a modularidade
        fastGreedy = compare_communities(community[i],graphs[i].community_fastgreedy().as_clustering(numberClasses).membership,method="rand")
        randIndex[1] += fastGreedy
        
        # usa o metodo de (label propagation method of Raghavan et al)
        labelPropag = compare_communities(community[i],graphs[i].community_label_propagation().membership,method="rand")
        randIndex[2] += labelPropag
        
        #Newman's leading eigenvector
        leadingEigen = compare_communities(community[i],graphs[i].community_leading_eigenvector(numberClasses).membership,method="rand")
        randIndex[3] += leadingEigen
        
        #baseado no algoritmo de Blondel et al.
        multilevel = compare_communities(community[i],graphs[i].community_multilevel().membership,method="rand")
        randIndex[4] += multilevel
        
        #baseado em random walks, usa o metodo de  Latapy & Pons
        walktrap = compare_communities(community[i],graphs[i].community_walktrap().as_clustering(numberClasses).membership,method="rand")
        randIndex[5] += walktrap

        kmeans += solveWithKMeans(list[i], community[i], numberClasses)        
    
    randIndex[6] = kmeans

    for i in range(7):
        randIndex[i]/=len(graphs)*1.0
    

    return randIndex

files = sorted(glob("./*.arff"))

label = []
data = []
x = []
kmeans = 0.0

label.append("Edge Betweenness")
label.append("Fast Greedy")
label.append("Label Propagation")
label.append("Leading Eigenvector")
label.append("Multilevel")
label.append("Walktrap")
label.append("K-Means")

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
    print "%d: cria os grafos" % (k)
    connectedGraphs = []
    connectedLists  = []
    connectedGraphs,connectedLists = testGraphs(graphs)
    
    print len(connectedGraphs)
    if(len(connectedGraphs) >= 8):
        randIndex = solve(connectedGraphs,connectedLists,communities,kmeans)
        data.append(randIndex)
        print randIndex[6]
        x.append(k)      
        print "Acabou!"


if(len(data) > 0):
    for i in range(len(data[0])):
        y = []
        
        for j in range(len(data)):
            y.append(data[j][i])
        print label[i]
        print y
        plt.plot(x,y,label=label[i])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.1)
    plt.legend()
    plt.show()
