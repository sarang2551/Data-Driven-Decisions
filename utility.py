from collections.abc import Callable, Iterable, Mapping
import heapq
from typing import Any
from scipy.stats import bernoulli
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, make_scorer
import concurrent.futures
import threading
def make_data(n, degree = 1,e = 0.5, d = 40, p = 5):
    b = bernoulli(0.5)
    # Construct a x feature vector
    I_p = np.identity(p)
    x = np.random.multivariate_normal(mean=np.array([0]*p),cov=I_p,size=(n))
    for i in range(n):
        for j in range(p):
            x[i][j] = abs(x[i][j])
    B = np.zeros(shape=(d,p))
    c = np.zeros(shape=(n,d))
    for i in range(d):
        for j in range(p):
            B[i][j] = b.rvs()
    
    # Construct a cost vector with dimension d
    for i in range(n):
        x_i = x[i].reshape(p,1)
        for j in range(d):
            noise = np.random.uniform(low=1-e,high=1+e)
            frac = 1/np.sqrt(p)
            vec_mult = (B @ x_i)[j]
            c[i][j] = (((frac*vec_mult + 3)**degree)+1)*noise
    return x,c

# make a graph with 40 edges
def make_graph(vec,n=5):
    # adjancency list
    graph = {}
    left_dif = -n
    right_dif = -(n-1)
    up_diff = -n
    down_diff = 0
    for node in range(n*n):
        if node not in graph:
            graph[node] = []
        # add all the left neighbours
        if node%n != 0:
            if node%n == 1:
                left_dif += (n-1)
            lidx = node+left_dif
            graph[node].append((node-1,vec[lidx]))
        # add all right neighbours
        if (node+1)%n != 0:
            if node%n == 0:
                right_dif += (n-1)
            ridx = node+right_dif
            graph[node].append((node+1,vec[ridx]))
        # add all up neighbours
        if node > (n-1):
            if node%n == 0:
                up_diff += (n-1)
            uidx = node+up_diff
            graph[node].append((node-n,vec[uidx]))
        # add all down neighbours
        if node < (n*n)-n:
            if node%n == 0:
                down_diff += (n-1)
            didx = node+down_diff
            graph[node].append((node+n,vec[didx]))
    return graph


class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxsize
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
	
    def __lt__(self,other):
	    return self.get_distance() < other.get_distance()


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


def dijkstra(aGraph, start, target):
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                # print 'updated : current = %s next = %s new_dist = %s' \
                #         %(current.get_id(), next.get_id(), next.get_distance())
				# else
                # print 'not updated : current = %s next = %s new_dist = %s' \
                #         %(current.get_id(), next.get_id(), next.get_distance())
        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)

def makeBinaryVector(path,special_cases=[0,1,2,3]):
    output = np.array([0]*40) 
    for i in range(1,len(path)):
        node = int(path[i])
        prev = int(path[i-1])
        diff = node-prev
        if diff == 1: # move right
            if prev in special_cases:
                right_idx = prev
                output[right_idx] = 1
            else:
                right_idx = (prev//5)*4 + prev
                output[right_idx] = 1
        elif diff == 5: # move down
            down_idx = ((prev//5)+1)*4 + prev
            output[down_idx] = 1
        elif diff == -1: # move left
            if node in special_cases:
                left_idx = node
                output[left_idx] = left_idx
            else:
                left_idx = (node//5)*4 + node
                output[left_idx] = 1
    return output



def get_shortest_path(g:dict):
	graph = Graph()
	for ii in range(len(g.keys())):
		graph.add_vertex(str(ii))
	for i in range(len(g.keys())):
		neighbours = g[i]
		for neigh,weight in neighbours:
			graph.add_edge(str(i),str(neigh),weight)
	# for v in graph:
	# 	for w in v.get_connections():
	# 		vid = v.get_id()
	# 		wid = w.get_id()
	# 		print(f"{vid} --> {wid} --> {v.get_weight(w)}")
	dijkstra(graph, graph.get_vertex("0"), graph.get_vertex(f"{len(g.keys())-1}")) 
	target = graph.get_vertex(f"{len(g.keys())-1}")
	path = [target.get_id()]
	shortest(target,path)
	return path[::-1]
    
def run_concurrent_tasks(tasks,params):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(t,*params) for t in tasks]
        return (f.result() for f in futures)

def makeHist(cost,title,labelX):
    sns.histplot(cost,bins=np.arange(20,80,2))
    plt.xlabel(labelX)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()
def makeHists(mss,acpp,title):
    plt.figure(figsize=(8,6))
    plt.hist(mss,alpha=0.3,bins=np.arange(20,80,2))
    plt.hist(acpp,alpha=0.5,color='orange',bins=np.arange(20,80,2))
    plt.legend(['MSS Cost','ACPP Cost'])
    plt.title(title)
    plt.show()
def makeScatter(mss,acpp,title):
    left = int(min(min(acpp),min(mss))) - 5
    right = int(max(max(acpp),max(mss))) + 5
    sns.scatterplot(x=acpp,y=mss)
    plt.plot(range(left,right),range(left,right),c="orange")
    plt.xlim((left,right))
    plt.ylim((left,right))
    plt.xlabel("ACPP Costs")
    plt.ylabel("MSS Costs")
    plt.title(title)
    plt.show()

def acpp_optimization(y_true,y_pred):
    actual_cost = 0
    for idx in range(len(y_true)):
        predictedVec = np.array(y_pred[idx]).reshape(40,1)
        g = make_graph(predictedVec)
        shortestPredictedPath = np.array(makeBinaryVector(get_shortest_path(g))).reshape(40,1)

        #optimumVec = np.array(y_true[idx]).reshape(40,1)
        #optimumShortestPath = np.array(makeBinaryVector(get_shortest_path(make_graph(optimumVec)))).reshape(40,1)

        # calculate actual cost * shortest predicted path
        actual_cost += y_true[idx]@shortestPredictedPath
        #optimum_cost = y_true[idx]@optimumShortestPath
        #difference += actual_cost-optimum_cost
    return actual_cost

def MSE():
    return make_scorer(mean_squared_error, greater_is_better=False)
def ACPP():
    return make_scorer(acpp_optimization,greater_is_better=False)

def price_volume_correlation(x_data:pd.DataFrame,y_data:pd.DataFrame,num_assests,xname,aggregate=True):
    # correlation of the X stock with all the y assests (averaged out?)
    # returns a pd.Series object with shape (n,1)
    volume_x = x_data['Volume']
    temp_arr = np.zeros(shape=(len(x_data),num_assests))
    for i in range(num_assests):
        price_y = y_data[y_data.columns[i]]
        for j in range(len(temp_arr)):
            x_vec = volume_x[:j+1]
            y_vec = price_y[:j+1]
            correlation = y_vec.corr(x_vec)
            temp_arr[j][i] = correlation if str(correlation) != "nan" else 0
    if aggregate:
        return pd.DataFrame(data=temp_arr.mean(axis=1),index=y_data.index,columns=[f'price_volume_correlation_{xname}'])
    else:
         return pd.DataFrame(data=temp_arr,index=y_data.index,columns=[f'price_volume_correlation_{xname}'])

def return_volume_correlation(X:pd.DataFrame,y:pd.DataFrame,x_data,xname,num_assests,aggregate=True):
    # correlation of the returns from stock y with the volume of stock x
    volume_x = X['Volume']
    temp_arr = np.zeros(shape=(len(x_data),num_assests))
    for i in range(num_assests):
        return_arr = y.iloc[:,i].pct_change().fillna(0)
        for j in range(len(temp_arr)):
              x_vec = volume_x[:j+1]
              y_vec = return_arr[:j+1]
              correlation = y_vec.corr(x_vec)
              temp_arr[j][i] = correlation if str(correlation) != "nan" else 0
    if aggregate:
        return pd.DataFrame(data=temp_arr.mean(axis=1),index=y.index,columns=[f'return_volume_correlation_{xname}'])
    else:
         return pd.DataFrame(data=temp_arr,index=y.index,columns=[f'return_volume_correlation_{xname}'])