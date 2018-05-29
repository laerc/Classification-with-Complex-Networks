from matplotlib.pyplot import *

def comparator(key):
    return key[1]

list = []
edgesList = []
graph = []
k=1

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


numberVertices = len(list)

#calcula a distancia euclidiana dos pontos e monta uma lista
for i in range(len(list)):
    edgesList.append([])
    for j in range(len(list)):
        if i == j:
            continue
        d = sum([ (list[i][x]-list[j][x])*(list[i][x]-list[j][x]) for x in range(2)])
        edgesList[i].append([j, d])
    edgesList[i] = sorted(edgesList[i],key=comparator)

ok = 0
#Usando o algoritmo knn, pegamos os k vertices mais proximos de u para se conectar.
for i in range(numberVertices):
	if(i <= 4):
		scatter(list[i][0],list[i][1],c='blue',s=50)
	else:
		scatter(list[i][0],list[i][1],c='red',s=50)

	for j in range(k):
		u,_ = edgesList[i][j]	
		if(ok == 0):
			plot([list[u][0],list[i][0]],[list[u][1],list[i][1]],c='black',LineWidth=2)
		ok = 
1
'''
scatter(-0.014123,-0.081213,c='blue',s=50)
plot([-0.014123,-0.115522],[-0.081213,-0.056579],c='black',LineWidth=3)
scatter(-0.115522,-0.056579,c='blue',s=50)
scatter(-0.226233,-0.039120,c='blue',s=50)
scatter(0.348492,0.103122,c='blue',s=50)
scatter(-1.277988,0.464366,c='blue',s=50)


scatter(1.120817,1.472072,c='red',s=50)
scatter(-0.758295,-1.920877,c='red',s=50)
scatter(0.445976,-0.910376,c='red',s=50)
scatter(-0.458929,-1.807199,c='red',s=50)
scatter(-0.405391,-1.841404,c='red',s=50)
'''

xlabel("Feature 1");
ylabel("Feature 2");

show()