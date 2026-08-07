#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import matplotlib.pyplot as plt
import pandas as pd
import matrixModification as mm
def OneFloorIter(N, d, M, gamma):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    if(gamma != 1.00):
        demand = M*gamma #scale down M (demandMatrix) by factor theta (gamma)
    else: 
        demand = np.array(M)
    linkCapacity = np.floor(demand)



    outDegree = [0]*N # How many outgoing links each node has left
    inDegree = [0]*N # How many incoming links each node has left
    for row in range(N):
        outDegree[row] = int(np.min([d,N-1]) - np.sum(linkCapacity[row]))
    for column in range(N):
        inDegree[column] = int(np.min([d,N-1]) - np.sum(linkCapacity[:,column]))


    dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 


    outLeft = [ x - dRes for x in outDegree]
    inLeft = [ x - dRes for x in inDegree]

    
    G = nx.random_regular_graph(dRes,N)
    fct.addGraphToMatrix(G, linkCapacity)
    fct.match_and_increment(outLeft, inLeft, linkCapacity)
    res = fct.thetaEdgeFormulation(linkCapacity, M, N, input_graph=False)
    return res


if __name__ == "__main__":
    matrixname= "skew-16-0.2"
    N= 16
    dE = 14
    workdir="./matrices/"
    demandMatrix = np.loadtxt(workdir+matrixname+".mat", usecols=range(N))
    demandMatrix = mm.Sinkhorn_Knopp(demandMatrix)
    demandMatrix = demandMatrix * dE
    print(fct.plotGamma(N, dE, demandMatrix, matrixname))


    # plotGamma(8,dE, demandMatrix)

    # findThroughput(16,14,matrixname, "Optimal")

    

    # print(OneFloorIter(8, dE, demandMatrix, 0.7))

    # end = time.time()
    # length = end - start
    # print(length)

