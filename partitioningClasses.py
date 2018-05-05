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

#Below we see all the functions

#Comparator to a sort
def comparator(key):
    return key[1]

def parseData(fileName):
    
    list = []
    listKMeans = []
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

    return [community,list]
    

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
        list[i][numberFeatures] = int(list[i][numberFeatures])
   
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
        
def solveGraphs(entries, graphs, community, metric_method):
    
    numberClasses = community[-1][-1]+1    
    ret = {}
    
    for x in entries:
        ret[x] = 0.0

    for i in range(len(graphs)):    
        if entries['edgeBetweeness'] == True:
            #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
            edgeBetweeness = compare_communities(community[i],graphs[i].community_edge_betweenness(clusters=numberClasses).as_clustering(numberClasses).membership,method=metric_method)
            ret['edgeBetweeness'] += edgeBetweeness 

        if entries['fastGreedy'] == True:
            #vai aglomerando as comunidades ate que nao aumenta a modularidade
            fastGreedy = compare_communities(community[i],graphs[i].community_fastgreedy().as_clustering(numberClasses).membership,method=metric_method)
            ret['fastGreedy'] += fastGreedy
        
        if entries['labelPropag'] == True:
            # usa o metodo de (label propagation method of Raghavan et al)
            labelPropag = compare_communities(community[i],graphs[i].community_label_propagation().membership,method=metric_method)
            ret['labelPropag'] += labelPropag
            
        if entries['leadingEigen'] == True:
            #Newman's leading eigenvector
            leadingEigen = compare_communities(community[i],graphs[i].community_leading_eigenvector(numberClasses).membership,method=metric_method)
            ret['leadingEigen'] += leadingEigen
            
        if entries['multilevel'] == True:    
            #baseado no algoritmo de Blondel et al.
            multilevel = compare_communities(community[i],graphs[i].community_multilevel().membership,method=metric_method)
            ret['multilevel'] += multilevel
            
        if entries['walktrap'] == True:
            #baseado em random walks, usa o metodo de  Latapy & Pons
            walktrap = compare_communities(community[i],graphs[i].community_walktrap().as_clustering(numberClasses).membership,method=metric_method)
            ret['walktrap'] += walktrap

        if entries['infoMap'] == True:
            #verify how to take the membership
            infoMap = compare_communities(community[i],graphs[i].community_infomap().membership,method=metric_method)
            ret['infoMap'] += infoMap  

    for i in entries:
        if(entries[i] == True):
            ret[i]/=len(graphs)*1.0
    

    return ret

def solveIAMethods(list, community, methods):
    
    ret = []
    listKMeans = []

    numberClasses = community[-1][-1]+1    

    for i in range(len(methods)):
        ret.append(0.0)

    for i in range(len(list)):
        for j in range(len(methods)):
            ret[j]+=solveWithKMeans(list[i], community[i], numberClasses, methods[j])

    return ret


#This method solves the problem with the algorithm k-Means
def solveWithKMeans(list, community, numberClasses, method):
    #print len(list[0]), len(community), numberClasses, method
    
    solve = KMeans(n_clusters = numberClasses, random_state = 0).fit(list)
    kmeansTmp = compare_communities(solve.labels_,community,method = method)
    
    return kmeansTmp
