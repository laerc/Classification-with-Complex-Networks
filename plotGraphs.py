from matplotlib.pyplot import *

def comparator(key):
    return key[1]

list = []
edgesList = []
graph = []
k=1
list.append([-0.014123,-0.081213,0])
list.append([-0.115522,-0.056579,0])
list.append([-0.226233,-0.039120,0])
list.append([0.348492,0.103122,0])
list.append([-1.277988,0.464366,0])
list.append([1.120817,1.472072,1])
list.append([-0.758295,-1.920877,1])
list.append([0.445976,-0.910376,1])
list.append([-0.458929,-1.807199,1])
list.append([-0.405391,-1.841404,1])

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