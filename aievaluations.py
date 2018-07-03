from partitioningClasses import *

def EvaluateKMeans(files):
    numFiles = 0

    kmeans = {'nmi' : 0, 'rand' : 0}
    dataList = []
    communityList = []
    numberClasses = 0

    for file in files:
        numFiles += 1
        y,x = parseData(file)

        #remove the last colugnm
        for i in range(len(y)):
            numberClasses = int(x[i][-1])+1
            x[i].pop()
        
        dataList.append(x)
        communityList.append(y)
        # This means that we got all the files with the same parameters, but different values
        if(numFiles == 10):
            for i in range(len(dataList)):
                kmeans['nmi']  += solveWithKMeans(dataList[i],np.asarray(communityList[i]),numberClasses,"nmi")
                kmeans['rand'] += solveWithKMeans(dataList[i],np.asarray(communityList[i]),numberClasses,"rand")
            kmeans['nmi'] /=10.0
            kmeans['rand']/=10.0
            return (kmeans)

def EvaluateEM(files):
    numFiles = 0

    dataList = []
    communityList = []
    numberClasses = 0

    for file in files:
        numFiles += 1
        y,x = parseData(file)

        #remove the last element(that indicates the class label)
        for i in range(len(y)):
            numberClasses = int(x[i][-1]+1)
            x[i].pop()
        
        dataList.append(x)
        communityList.append(y)

        if(numFiles == 10):
            EM = {'nmi' : [], 'rand' : []}
            em_nmi_v  = []
            em_rand_v = []
            
            for j in range(len(dataList)):
                EM['nmi'].append(0)
                EM['rand'].append(0)
                
                for i in range(20):
                    em_nmi  = solveWithEM(dataList[j],communityList[j],numberClasses,'nmi')
                    em_rand = solveWithEM(dataList[j],communityList[j],numberClasses,'rand')
                    EM['nmi'][j]  += em_nmi
                    EM['rand'][j] += em_rand

                em_nmi_v.append(EM['nmi'][j]/20.0)
                em_rand_v.append(EM['rand'][j]/20.0)

            numFiles = 0
            communityList = []
            numFiles = 0
            return { 'nmi' : sum(em_nmi_v)/len(dataList), 'nmi_dp'  : np.std(em_nmi_v, dtype=np.float64),
                     'rand': sum(em_rand_v)/len(dataList), 'rand_dp': np.std(em_rand_v, dtype=np.float64) }