import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from glob import *
from igraph import *
from partitioningClasses import *
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator

import cv2 as cv

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
        #print method
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
    best_performance_nmi = {}
    best_performance_rand = {}
    label = {}
    color = {}
    entries = {}
    x = []
    y = {}
    kmeans = 0.0

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

    entries['edgeBetweeness'] = True
    entries['fastGreedy'] = True
    entries['labelPropag'] = True
    entries['leadingEigen'] = True
    entries['multilevel'] = True
    entries['walktrap'] = True
    entries['infoMap'] = True
    entries['kmeans'] = True
    entries['EM'] = True
    y = {}
    
    y['nmi'] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : [], 'EM' : []}#, 'consensus' : []}

    y['rand'] = {'edgeBetweeness' : [], 'fastGreedy' : [], 'labelPropag' : [], 'leadingEigen' : [],
          'multilevel' : [], 'walktrap' : [], 'infoMap' : [], 'kmeans' : [], 'EM' : []}#, 'consensus' : []}

    #print ("Executing these files : %s" % (files))

    for k in range(2,maxk):
        communities = []
        graphs = []    
        kmeans = 0.0
        #print k
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

        #for i in range(len(connectedLists)):
        #    for j in range(len(connectedLists[i])):
        #        connectedLists[i][j].pop()

        if(len(connectedGraphs) >= 8):
            # use rand for rand index and nmi for nmi clustering evaluation
            cur_val_nmi = 0.0
            cur_val_rand = 0.0

            ai_methods = solveIAMethods(connectedLists, communities, methods=['nmi', 'rand'])
            ret = solveGraphs(entries, connectedGraphs, communities, metric_method=method)

            for key_method, value_method in ai_methods.iteritems():
                for key_algo, value_algo in value_method.iteritems():
                    y[key_method][key_algo].append(value_algo/len(connectedLists))

            #ret_consensus = solveWithConsensus(connectedGraphs, communities, method, tol, np)
            
            '''print "\n--------------------------------------------------------\n"
            
            print k 
            ret_consensus = solveWithConsensus(connectedGraphs, communities, method, tol, np)
            print "\n--------------------------------------------------------\n"
            '''

            best_performance_nmi.update({k : ret['nmi']})
            best_performance_rand.update({k : ret['rand']})

            for key, value in ret['nmi'].iteritems():
                y['nmi'][key].append(sum(value)/len(connectedGraphs)*1.0)
                cur_val_nmi = max(cur_val_nmi, sum(value)/len(connectedGraphs)*1.0)

            for key, value in ret['rand'].iteritems():
                y['rand'][key].append(sum(value)/len(connectedGraphs)*1.0)
                cur_val_rand = max(cur_val_rand, sum(value)/len(connectedGraphs)*1.0)

            x.append(k)

            if(cur_val_nmi < maxi_eval_nmi and cur_val_rand < maxi_eval_rand and k >= 10):
                break
            maxi_eval_nmi = max(maxi_eval_nmi,cur_val_nmi)
            maxi_eval_rand = max(maxi_eval_rand,cur_val_rand)


    ranks = findBestRank(best_performance_nmi)
    '''plotRank(ranks, "NMI")'''
    #print ("NMI")
    print "nmi"
    print ranks
    
    ranks = findBestRank(best_performance_rand)
    print "rand"
    print ranks

    print x
    print y
    #plotRank(ranks, "Rand Index")
    '''print ("Rand Index")
    print (ranks)
    print "\n"
    print x
    print "\n"    
    print y
    print "\n"
    print "------------------------------------------------------\n\n"
    '''
    '''
    plt.style.use('fivethirtyeight')

    for key, val in y['nmi'].iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("NMI(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()

    for key, val in y['rand'].iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()
    '''

def plot(ranks_nmi, ranks_rand, x, y):
    plt.style.use('fivethirtyeight')
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))

    color = {}
    label = {}
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

    plt.xticks(rotation=45)
    plotRank(ranks_nmi , "NMI")
    plt.xticks(rotation=45)
    plotRank(ranks_rand, "Rand Index")
    

    plt.style.use('fivethirtyeight')

    for key, val in y['nmi'].iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("NMI(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()

    for key, val in y['rand'].iteritems():
        plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()

'''
ranks_nmi  = {'labelPropag': 6.0, 'fastGreedy': 1.2, 'walktrap': 1.7, 'leadingEigen': 2.3, 'infoMap': 4.5, 'multilevel': 5.0, 'edgeBetweeness': 1.5}
ranks_rand = {'labelPropag': 5.6, 'fastGreedy': 1.2, 'walktrap': 1.7, 'leadingEigen': 2.2, 'infoMap': 4.6, 'multilevel': 5.6, 'edgeBetweeness': 1.5}

x = [3, 4, 5, 6, 7, 8, 9, 10]
y = {'rand': {'EM': [0.5015555555555556, 0.5037979797979798, 0.5036565656565657, 0.49923232323232325, 0.501090909090909, 0.5005656565656567, 0.503070707070707, 0.5018181818181818], 'infoMap': [0.5790505050505048, 0.5979999999999999, 0.6183030303030304, 0.6311717171717172, 0.647111111111111, 0.6770707070707069, 0.7152525252525253, 0.7368888888888888], 'multilevel': [0.6264444444444444, 0.6393333333333332, 0.6485252525252524, 0.6648686868686867, 0.6731717171717171, 0.6862626262626261, 0.6868484848484847, 0.6867878787878787], 'labelPropag': [0.5836161616161617, 0.6061414141414141, 0.643090909090909, 0.6482424242424242, 0.6484040404040403, 0.6672525252525252, 0.6654545454545453, 0.6102424242424244], 'fastGreedy': [0.9186060606060605, 0.9255555555555557, 0.9709696969696969, 0.986121212121212, 0.9536767676767678, 0.9753737373737372, 0.9651919191919192, 0.9191111111111111], 'edgeBetweeness': [0.9652525252525253, 0.9141414141414141, 0.926888888888889, 0.859030303030303, 0.8658383838383837, 0.8576363636363636, 0.848181818181818, 0.6786666666666665], 'walktrap': [0.9213737373737374, 0.9288080808080809, 0.9120202020202021, 0.9572121212121212, 0.9620606060606061, 0.9667878787878788, 0.935070707070707, 0.8750101010101012], 'leadingEigen': [0.6373939393939393, 0.8383434343434344, 0.9204646464646465, 0.9069292929292929, 0.8895757575757577, 0.8929292929292929, 0.9058787878787878, 0.8618181818181817], 'kmeans': [0.5004848484848485, 0.5004848484848485, 0.5004848484848485, 0.5004848484848485, 0.5004848484848485, 0.5004848484848485, 0.5004848484848485, 0.5004848484848485]}, 'nmi': {'EM': [0.006064042038705345, 0.0055902347539386315, 0.01146029986327975, 0.01074623022623163, 0.01788617908432803, 0.01604878456548018, 0.03586449964891239, 0.008418011313691983], 'infoMap': [0.4186906449213098, 0.44653463534389537, 0.4698299562224186, 0.48956272840715015, 0.5006130486894588, 0.5481605410920851, 0.5709169080045136, 0.5711256904402946], 'multilevel': [0.48673810299749354, 0.49564527938961983, 0.5053820029144855, 0.5308236236868485, 0.5365041690562719, 0.5607393062082824, 0.5665934258721499, 0.5462091938222964], 'labelPropag': [0.4113847140145552, 0.4378064378674174, 0.47370609515815965, 0.48661895991162096, 0.45829263297725253, 0.4519675117038823, 0.4213686558734544, 0.2795344050844284], 'fastGreedy': [0.8220291364794005, 0.8257047325972318, 0.9191520357699714, 0.9544097421864344, 0.8903536049875262, 0.9395770601617013, 0.9145593197221169, 0.800768049641864], 'edgeBetweeness': [0.9088379492713237, 0.8161747221618457, 0.841968753425037, 0.7318965450114725, 0.735209760162196, 0.7135037806376036, 0.6990759161925959, 0.43332397771667497], 'walktrap': [0.8313624755344433, 0.8463130245818485, 0.8100327174842393, 0.905110570804007, 0.900500521685211, 0.9194672377235877, 0.834420900736385, 0.7441642031535847], 'leadingEigen': [0.27815550283746315, 0.6227517307335422, 0.8030043997848948, 0.7766919711405482, 0.7498885953587469, 0.7422280016407361, 0.7703574258234591, 0.6809796980219136], 'kmeans': [0.008539442881051011, 0.008539442881051011, 0.008539442881051011, 0.008539442881051011, 0.008539442881051011, 0.008539442881051011, 0.008539442881051011, 0.008539442881051011]}}

plot(ranks_nmi,ranks_rand,x,y)
'''

files = sorted(glob("./*.arff"))

#EvaluateKMeans(files,["nmi", "rand"])
for i in range(0,len(files),10):
    cur_files = []
    
    for j in range(i,i+10):
        cur_files.append(files[j])
    
    print cur_files[0]
    main(cur_files)

