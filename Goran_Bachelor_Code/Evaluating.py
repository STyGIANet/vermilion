import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import random_Matrices as rM
import Throughput_as_Function as fct
import Floor_Draft as fd
import Rounding_Draft as rd
import sys

def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "random-skew-"
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-"
    ]
    for i in range(4):
        res[i] += str(N)
    for j in range(9):
        print(j+4)
        res[j+4] += str(N) + "-0." + str(j+1)
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap2","heatmap3", "topoopt"]

    


if __name__ == "__main__":
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    n_values_used = [16]
    matrices16 = generate_synthmatrix_names(16) + organicmatrices16
    print(matrices16)
    N = 16
    dE = 14
    # floorResults =  []
    # roundingResults = []
    randomRes = []
    expanderRes =[]
    # CircleRes = []
    # ChordRes = []
    # ExpanderRes = []


 
    # print(np.array2string(matrix))
    # fct.preprocessing(matrix)
    # print(np.array2string(matrix))
    # matrix = matrix * dE
    # print(fd.thetaByFloor(N, dE, matrix, 5))
    # print(matrices16)
    # for matrix in matrices16:
    #     demandMatrix = np.loadtxt(workdir+matrix+".mat", usecols=range(N)) #TODO: Manually remove tiny entries
    #     # print(matrix + "__________________________________")
    #     # print(np.array2string(demandMatrix))
    #     # print("After filtering:___________________________________________")

    #     fct.filtering(demandMatrix)
    #     # print(np.array2string(demandMatrix))
    #     # print("After normalizing:_____________________________________________")
    #     demandMatrix = fct.return_normalized_matrix(demandMatrix)
    #     # print(np.array2string(demandMatrix))
    #     demand = demandMatrix *dE
    #     # ExpanderRes.append(fct.thetaEdgeFormulation(nx.random_regular_expander_graph(N, dE, max_tries= 10000 ),demand, N))
    #     # floorResults.append(fd.thetaByFloor(N,dE, demand, 6))
    #     roundingResults.append(rd.alternativeTheta(N,dE, demand, 6)) #Alternative Rounding!
    #     # CircleRes.append(fct.thetaEdgeFormulation(fct.createCircleGraph(N, dE), demand, N))
    #     # ChordRes.append(fct.thetaEdgeFormulation(fct.createPseudoChord(N,dE), demand, N))

    # # print("Results for each matrix using Floor Method:")
    # # for i in range(len(matrices16)):
    # #     print(matrices16[i] + ": " + str(floorResults[i]))
    # print("Results for each matrix using Rounding Method:")
    # for j in range(len(matrices16)):
    #     print(matrices16[j] + ": " + str(roundingResults[j]))
