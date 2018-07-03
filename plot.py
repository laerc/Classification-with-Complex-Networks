import Orange

def plot(rank_nmi, rank_rand, x, y):

    color = {}
    label = {}
    names = []
    avranks_nmi = []
    avranks_rand = []
    
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

    for key,val in label.items():
        names.append(val)
        avranks_nmi.append(rank_nmi[key])
        avranks_rand.append(rank_rand[key])

    cd = Orange.evaluation.compute_CD(avranks_nmi, 10) #tested on 10 datasets
    Orange.evaluation.graph_ranks(avranks_nmi, names, cd=cd, width=6, textspace=1.2)
    plt.show()

    cd = Orange.evaluation.compute_CD(avranks_rand, 10) #tested on 10 datasets
    Orange.evaluation.graph_ranks(avranks_rand, names, cd=cd, width=6, textspace=1.2)
    plt.show()
    
    plt.style.use('fivethirtyeight')

    for key, val in y['nmi'].items():
        if(key not in label):
            continue
        if(key == 'EM' or key == 'kmeans'):
            plt.plot(x,val,label=label[key],color=color[key], dashes=[5, 3])
        else:
            plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("NMI(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0.0,0.0,1.3,1])
    
    plt.show()

    for key, val in y['rand'].items():
        if(key == 'EM' or key == 'kmeans'):
            plt.plot(x,val,label=label[key],color=color[key], dashes=[5, 3])
        else:
            plt.plot(x,val,label=label[key],color=color[key])

    plt.xlabel("k")
    plt.ylabel("Rand Index(k)")
    plt.ylim(ymax=1.0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0.0,0.0,1.3,1])
    plt.show()
