import cv2 as cv
import numpy as np

from igraph import *
from sklearn.cluster import KMeans

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

#this method solves the problem with the algorithm Expectation Maximization
def solveWithEM(list, community, numberClasses, method):
    em = cv.ml.EM_create()
    em.setClustersNumber(numberClasses)
    _,_,clusters,_ = em.trainEM(np.asarray(list))
    em_eval = compare_communities([elem[0] for elem in clusters.tolist()], community, method=method)

    return em_eval
