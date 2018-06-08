# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:57:41 2017
@author: laercio
"""

from glob import *
from igraph import *

from igraph import arpack_options
from sklearn.cluster import KMeans

import cv2 as cv
import numpy as np
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
        list[i].pop()
   
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
    ret = {'nmi' : {}, 'rand' : {}}
    
    ret['nmi'] = ({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
            'multilevel' : [], 'walktrap' : [], 'infoMap' : []})

    ret['rand'] = ({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
            'multilevel' : [], 'walktrap' : [], 'infoMap' : []})

    for i in range(len(graphs)):    
        if entries['edgeBetweeness'] == True:
            #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
            edgeBetweeness = graphs[i].community_edge_betweenness(clusters=numberClasses).as_clustering(numberClasses).membership
            edgeBetweeness_nmi = compare_communities(community[i],edgeBetweeness,method="nmi")
            edgeBetweeness_rand = compare_communities(community[i],edgeBetweeness,method="rand")
            ret['nmi']['edgeBetweeness'].append(edgeBetweeness_nmi)
            ret['rand']['edgeBetweeness'].append(edgeBetweeness_rand)

        if entries['fastGreedy'] == True:
            #vai aglomerando as comunidades ate que nao aumenta a modularidade
            fastGreedy = graphs[i].community_fastgreedy().as_clustering(numberClasses).membership
            fastGreedy_nmi = compare_communities(community[i],fastGreedy,method="nmi")
            fastGreedy_rand = compare_communities(community[i],fastGreedy,method="rand")
            ret['nmi']['fastGreedy'].append(fastGreedy_nmi)
            ret['rand']['fastGreedy'].append(fastGreedy_rand)
        
        if entries['labelPropag'] == True:
            # usa o metodo de (label propagation method of Raghavan et al)
            labelPropag = graphs[i].community_label_propagation().membership
            labelPropag_nmi = compare_communities(community[i],labelPropag,method="nmi")
            labelPropag_rand = compare_communities(community[i],labelPropag,method="rand")
            ret['nmi']['labelPropag'].append(labelPropag_nmi)
            ret['rand']['labelPropag'].append(labelPropag_rand)
            
        if entries['leadingEigen'] == True:
            #Newman's leading eigenvector
            arpack_options.maxiter = 50000
            try:
                leadingEigen = graphs[i].community_leading_eigenvector(numberClasses).membership
                leadingEigen_nmi = compare_communities(community[i],leadingEigen,method="nmi")
                leadingEigen_rand = compare_communities(community[i],leadingEigen,method="rand")
                
            except:
                leadingEigen_nmi = 0.0
                leadingEigen_rand = 0.0

            ret['nmi']['leadingEigen'].append(leadingEigen_nmi)
            ret['rand']['leadingEigen'].append(leadingEigen_rand)
        
        if entries['multilevel'] == True:    
            #baseado no algoritmo de Blondel et al.
            multilevel = graphs[i].community_multilevel().membership
            multilevel_nmi = compare_communities(community[i],multilevel,method="nmi")
            multilevel_rand = compare_communities(community[i],multilevel,method="rand")
            ret['nmi']['multilevel'].append(multilevel_nmi)
            ret['rand']['multilevel'].append(multilevel_rand)
            
        if entries['walktrap'] == True:
            #baseado em random walks, usa o metodo de  Latapy & Pons
            walktrap = graphs[i].community_walktrap().as_clustering(numberClasses).membership
            walktrap_nmi = compare_communities(community[i],walktrap,method="nmi")
            walktrap_rand = compare_communities(community[i],walktrap,method="rand")
            ret['nmi']['walktrap'].append(walktrap_nmi)
            ret['rand']['walktrap'].append(walktrap_rand)

        if entries['infoMap'] == True:
            #verify how to take the membership
            infoMap = graphs[i].community_infomap().membership
            infoMap_nmi = compare_communities(community[i],infoMap,method="nmi")
            infoMap_rand = compare_communities(community[i],infoMap,method="rand")
            ret['nmi']['infoMap'].append(infoMap_nmi)
            ret['rand']['infoMap'].append(infoMap_rand)  

        #layout = graphs[i].layout("kk")
        #plot(graphs[i], mark_groups = True, layout=layout)
        #plot(graphs[i].community_fastgreedy().as_clustering(numberClasses), mark_groups = True, layout=layout)
        #return 

        #layout = graphs[i].layout("kk")
        #plot(graphs[i], layout = layout)
        #return

    return ret

def createConsensusGraphsWeighted(clusters, size, threshold, np):
    
    seen_before = set()
    adj_mat = [[0 for ii in range(size)] for jj in range(size)]
    
    #print adj_mat

    for key_i, val_i in clusters.items():
        seen_before.add(key_i)
        for key_j, val_j in clusters.items():
            if(key_j in seen_before): 
                continue
            
            if(len(val_i) == 0 or len(val_j) == 0):
                continue

            for i in range(size):
                for j in range(i+1,size):
                    
                    if val_i[0][i] == val_j[0][j]:
                        adj_mat[i][j] += 1

    #create a new graph
    #graph = Graph.Weighted_Adjacency(adj_mat,attr="weight",mode=ADJ_UNDIRECTED)
    #graph.add_vertices(size)
    #graph.es["weight"] = 1.0

    #for i in range(size):
    #    for j in range(i+1,size):
    #        if(adj_mat[i][j] >= threshold*size):
    #            graph.add_edge(i,j,weight=adj_mat[i][j])
                #print (graph[i,j])

    #print size
    #graph.simplify()
    return adj_mat

def solveWithConsensus(graphs, communities, metric_method, threshold, np):
    numberClasses = communities[-1][-1]+1
    results = {"nmi" : {}, "rand" : {}}   
    best_result = {"nmi" : {}, "rand" : {}}

    results["nmi"]  = ({'before' : [], 'after' : []})

    results["rand"] = ({'before' : [], 'after' : []})

    best_result["nmi"] = ({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 
        'leadingEigen' : [], 'multilevel' : [], 'walktrap' : [], 'infoMap' : []})

    best_result["rand"] = ({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 
        'leadingEigen' : [], 'multilevel' : [], 'walktrap' : [], 'infoMap' : []})

    for i in range(len(graphs)):
        ok = False
        after_maxi_nmi = 0.0
        after_maxi_rand = 0.0
        before_maxi_nmi = 0.0
        before_maxi_rand = 0.0
        cur_result = {"nmi" : {}, "rand" : {}}
        cur_result["nmi"] = ({'edgeBetweeness' : [0.0, 0], 'fastGreedy' : [0.0, 0], 'labelPropag' : [0.0, 0], 
            'leadingEigen' : [0.0, 0], 'multilevel' : [0.0, 0], 'walktrap' : [0.0, 0], 'infoMap' : [0.0, 0]})

        cur_result["rand"] = ({'edgeBetweeness' : [0.0, 0], 'fastGreedy' : [0.0, 0], 'labelPropag' : [0.0, 0], 
            'leadingEigen' : [0.0, 0], 'multilevel' : [0.0, 0], 'walktrap' : [0.0, 0], 'infoMap' : [0.0, 0]})

        k = 0
        j = 0

        graphs[i].es["weight"] = 1.0
        
        while(ok == False and j < 10):
            ok = True
            ret = []    
            j+=1

            ret.append({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
            'multilevel' : [], 'walktrap' : [], 'infoMap' : []})
            # usa o metodo de (label propagation method of Raghavan et al)
            
            labelPropag = graphs[i].community_label_propagation(weights="weight").membership
            ret[k]['labelPropag'].append(labelPropag)
            
            #Newman's leading eigenvector
            #leadingEigen = graphs[i].community_leading_eigenvector(weights="weight",clusters=numberClasses).membership
            #ret[k]['leadingEigen'].append(leadingEigen)
            
            #baseado no algoritmo de Blondel et al.
            multilevel = graphs[i].community_multilevel(weights="weight").membership
            ret[k]['multilevel'].append(multilevel)
            
            #baseado em random walks, usa o metodo de  Latapy & Pons
            walktrap = graphs[i].community_walktrap(weights="weight").as_clustering(numberClasses).membership
            ret[k]['walktrap'].append(walktrap)

            #verify how to take the membership
            infoMap = graphs[i].community_infomap(edge_weights="weight").membership
            ret[k]['infoMap'].append(infoMap)

            #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
            #edgeBetweeness = graphs[i].community_edge_betweenness(weights="weight",clusters=numberClasses).as_clustering(numberClasses).membership
            #ret[k]['edgeBetweeness'].append(edgeBetweeness)

            #vai aglomerando as comunidades ate que nao aumenta a modularidade
            fastGreedy = graphs[i].community_fastgreedy(weights="weight").as_clustering(numberClasses).membership
            ret[k]['fastGreedy'].append(fastGreedy)

            for key_j, val_j in ret[k].items():
                if(len(val_j) == 0):
                    continue
                #print val_j[0], communities[i], best_result["nmi"][key_j], key_j
                comp_commum_nmi  = compare_communities(communities[i], val_j[0], method="nmi")
                comp_commum_rand = compare_communities(communities[i], val_j[0], method="rand")
                if(j == 1):
                    before_maxi_nmi = max(before_maxi_nmi, comp_commum_nmi)
                    before_maxi_rand = max(before_maxi_rand, comp_commum_rand)

                if cur_result["nmi"][key_j][0] < comp_commum_nmi:
                    cur_result["nmi"][key_j][0] = comp_commum_nmi
                    cur_result["nmi"][key_j][1] = j
                
                if cur_result["rand"][key_j][0] < comp_commum_rand:
                    cur_result["rand"][key_j][0] = comp_commum_rand
                    cur_result["rand"][key_j][1] = j

            for key_i, val_i in ret[k].items():
                for key_j, val_j in ret[k].items():
                    if(len(val_i) == 0 or len(val_j) == 0):
                        continue
                    if(val_i[0] != val_j[0]):
                        ok = False
                        break

                if(ok == False):
                    break
            
            #They differ from at least one
            if(ok == False):
                adj_mat = createConsensusGraphsWeighted(ret[k], len(communities[i]), threshold, np)
                graphs[i] = Graph.Weighted_Adjacency(adj_mat,mode=ADJ_UNDIRECTED)
                #print adj_mat
            #They are all equal
            else:
                break

        for key_j, val_j in cur_result["nmi"].items():
            #print (key_j, val_j)
            after_maxi_nmi = max(after_maxi_nmi, val_j[0])
            best_result["nmi"][key_j].append(val_j[0])


        for key_j, val_j in cur_result["rand"].items():
            after_maxi_rand = max(after_maxi_nmi,val_j[0])
            best_result["rand"][key_j].append(val_j[0])

        #print ("best result after")
        results['nmi']['before'].append(before_maxi_nmi)
        results['nmi']['after'].append(after_maxi_nmi)
        results['rand']['before'].append(before_maxi_rand)
        results['rand']['after'].append(after_maxi_rand)
        #print (before_maxi_nmi, after_maxi_nmi, before_maxi_rand, after_maxi_rand)
        #print ("-----------------------------------------------------------------------")
            #print graphs[i].es["weight"]
            #layout = graphs[i].layout("kk")
            #plot(graphs[i], layout = layout)
            #return
        #print j

    return best_result, results

def createConsensusGraphs(clusters, threshold, np):
    D = {}
    graphs = {}

    for cluster in clusters:
        for key, val in cluster.items():
            if not(key in D):
                D[key] = [[0 for i in range(len(val[0]))] for j in range(len(val[0]))]

            for i in range(len(val[0])):
                for j in range(len(val[0])):
                    if val[0][i] == val[0][j]:
                        D[key][i][j] += 1

    for key, val in cluster.items():         
        graphs[key] = Graph.Weighted_Adjacency(D[key],mode=ADJ_UNDIRECTED)

    return graphs
'''
    for key, val in cluster.items():
        graph = Graph()
        graph.add_vertices(val[0])

        for i in range(len(D[key])):
            for j in range(len(D[key][i])):
                if(D[key][i][j] < threshold*np):
                    D[key][i][j] = 0
                else:
                    D[key][i][j] = 1
                    graph.add_edge(i,j)
                    graph.add_edge(j,i)

        graph.simplify()
        #graphs[key] = Graph.Adjacency(D[key])
        graphs[key] = graph
'''     

def createMatrixConsensuns(entries, graphs, np, threshold, numberClasses):
    ret = []
    D = []
    allequal = []

    for i in range(len(graphs)):
        allequal.append({'edgeBetweeness' : True, 'fastGreedy' : True, 'labelPropag' : True, 'leadingEigen' : True,
            'multilevel' : True, 'walktrap' : True, 'infoMap' : True })

    for i in range(len(graphs)):
        ret.append([])
        
        for k in range(np):    
            ret[i].append({'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
            'multilevel' : [], 'walktrap' : [], 'infoMap' : []})
            if entries['edgeBetweeness'] == True:
                #vai recalculando o betweenness, e as arestas de maior betweenness sao tiradas
                edgeBetweeness = graphs[i]['edgeBetweeness'].community_edge_betweenness(weights="weight",clusters=numberClasses).as_clustering(numberClasses).membership
                ret[i][k]['edgeBetweeness'].append(edgeBetweeness)
                allequal[i]['edgeBetweeness'] = allequal[i]['edgeBetweeness'] & isCommunitiesEqual(ret[i][k]['edgeBetweeness'], ret[i][k-1]['edgeBetweeness'])

            if entries['fastGreedy'] == True:
                #vai aglomerando as comunidades ate que nao aumenta a modularidade
                fastGreedy = graphs[i]['fastGreedy'].community_fastgreedy(weights="weight").as_clustering(numberClasses).membership
                ret[i][k]['fastGreedy'].append(fastGreedy)
                allequal[i]['fastGreedy'] = allequal[i]['fastGreedy'] & isCommunitiesEqual(ret[i][k]['fastGreedy'], ret[i][k-1]['fastGreedy'])
            
            if entries['labelPropag'] == True:
                # usa o metodo de (label propagation method of Raghavan et al)
                labelPropag = graphs[i]['labelPropag'].community_label_propagation(weights="weight").membership
                ret[i][k]['labelPropag'].append(labelPropag)
                allequal[i]['labelPropag'] = allequal[i]['labelPropag'] & isCommunitiesEqual(ret[i][k]['labelPropag'], ret[i][k-1]['labelPropag'])
                
            if entries['leadingEigen'] == True:
                #Newman's leading eigenvector
                leadingEigen = graphs[i]['leadingEigen'].community_leading_eigenvector(weights="weight",clusters=numberClasses).membership
                ret[i][k]['leadingEigen'].append(leadingEigen)
                allequal[i]['leadingEigen'] = allequal[i]['leadingEigen'] & isCommunitiesEqual(ret[i][k]['leadingEigen'], ret[i][k-1]['leadingEigen'])
                
            if entries['multilevel'] == True:    
                #baseado no algoritmo de Blondel et al.
                multilevel = graphs[i]['multilevel'].community_multilevel(weights="weight").membership
                ret[i][k]['multilevel'].append(multilevel)
                allequal[i]['multilevel'] = allequal[i]['multilevel'] & isCommunitiesEqual(ret[i][k]['multilevel'], ret[i][k-1]['multilevel'])
                
            if entries['walktrap'] == True:
                #baseado em random walks, usa o metodo de  Latapy & Pons
                walktrap = graphs[i]['walktrap'].community_walktrap(weights="weight").as_clustering(numberClasses).membership
                ret[i][k]['walktrap'].append(walktrap)
                allequal[i]['walktrap'] = allequal[i]['walktrap'] & isCommunitiesEqual(ret[i][k]['walktrap'], ret[i][k-1]['walktrap'])

            if entries['infoMap'] == True:
                #verify how to take the membership
                infoMap = graphs[i]['infoMap'].community_infomap(edge_weights="weight").membership
                ret[i][k]['infoMap'].append(infoMap)
                allequal[i]['infoMap'] = allequal[i]['infoMap'] & isCommunitiesEqual(ret[i][k]['infoMap'], ret[i][k-1]['infoMap'])

        D.append(createConsensusGraphs(ret[i], threshold, np))

    return D, allequal, ret


def consensus(entries, graphs, community, metric_method, np, threshold):
    numberClasses = community[-1][-1]+1
    current_graph = []
    max_iter = 100

    for i in range(len(graphs)):
        graphs[i].es["weight"] = 1.0
        current_graph.append({'edgeBetweeness' : graphs[i], 'fastGreedy' : graphs[i], 'labelPropag' : graphs[i], 
            'leadingEigen' : graphs[i], 'multilevel' : graphs[i], 'walktrap' : graphs[i], 'infoMap' : graphs[i]})

    for i in range(max_iter):
        print ("iteration %d" % i)
        runs, allequal, ret = createMatrixConsensuns(entries, current_graph, np, threshold, numberClasses)
        current_graph = runs

        if(consensusSolved(allequal)):
            return computeAllEqual(ret, community)

    return ret

def computeAllEqual(ret, community):
    results = {"nmi" : {}, "rand" : {}}

    results["nmi"] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 
            'leadingEigen' : [], 'multilevel' : [], 'walktrap' : [], 'infoMap' : []}

    results["rand"] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 
            'leadingEigen' : [], 'multilevel' : [], 'walktrap' : [], 'infoMap' : []}

    for i in range(len(ret)):
        for key, val in ret[i][0].items():
            results["nmi"][key].append(compare_communities(val[0],community[i],method = "nmi"))
            results["rand"][key].append(compare_communities(val[0],community[i],method = "rand"))

    return results

def isCommunitiesEqual(communityA, communityB):

    if communityA == communityB:
        return True
    else:
        return False

def consensusSolved(allequal):
    
    for i in range(len(allequal)):
        for key, val in allequal[i].items():
            if(val == False):
                return False

    return True

def solveIAMethods(list, community, methods):
    
    ret = {"nmi" : {}, "rand" : {}}
    ret["nmi"]  = {"kmeans" : 0.0, "EM" : 0.0}
    ret["rand"] = {"kmeans" : 0.0, "EM" : 0.0}

    numberClasses = community[-1][-1]+1    


    for i in range(len(list)):
        for key_method, _ in ret.items():
            ret[key_method]["kmeans"] += solveWithKMeans(list[i], community[i], numberClasses, key_method)
            ret[key_method]["EM"]     += solveWithEM    (list[i], community[i], numberClasses, key_method)

    return ret

#This method solves the problem with the algorithm k-Means
def solveWithKMeans(list, community, numberClasses, method):
    
    solve = KMeans(n_clusters = numberClasses, random_state = 0).fit(list)
    kmeansTmp = compare_communities(solve.labels_,community,method = method)
    
    return kmeansTmp

def solveWithEM(list, community, numberClasses, method):
    
    em = cv.ml.EM_create()
    em.setClustersNumber(numberClasses)
    _,_,clusters,_ = em.trainEM(np.asarray(list))
    em_eval = compare_communities([elem[0] for elem in clusters.tolist()], community, method=method)

    return em_eval
