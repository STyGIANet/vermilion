#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sept 10 01:14:20 2024

@author: vamsi
"""
import networkx as nx
import numpy as np
import math
import  random
import sys
from collections import defaultdict
# import gurobipy as gp
# from gurobipy import GRB
import copy

#%%

# This script generates topologies for All-to-All, Permutation, and Skew. For anyother patterns, please pass the demand matrix accordingly.

# nTors = int(sys.argv[1]) # n_t
# nServersPerTor = int(sys.argv[2]) # n_u, the number of uplinks, also the number of downlinks. No oversubscription at ToR.
# k = int(sys.argv[3])
# dm = str(sys.argv[4])
# skew = float(sys.argv[5])
# reconf = int(sys.argv[4]) # in microseconds

nTors = 8
nServersPerTor = 1
linkSpeed = 400 # in Gbps
propagation_delay = 0.5 # in microseconds

reconf = 0.5 # in microseconds
reconfStr = "0.5us"
guard_band = (1500*8/(linkSpeed))/1000 # in microseconds
# reconf = 0.010 # in microseconds
# reconfStr = "10ns"
# guard_band = (500*8/(linkSpeed))/1000 # in microseconds

assert(nTors%2==0)
# Require ToR count divisible by per-ToR servers so period = k*N/dE is integer.
assert(nTors % nServersPerTor == 0)

totalServers = nServersPerTor*nTors
nDownlinks = nServersPerTor
nUplinks = 4 # n_u

N=nTors
dE = nUplinks
k_s = [8]
""" matrices = [
            "chessboard-"+str(nTors),
            "uniform-"+str(nTors),
            "permutation-"+str(nTors),
            "skew-"+str(nTors)+"-0.1",
            "skew-"+str(nTors)+"-0.2",
            "skew-"+str(nTors)+"-0.3",
            "skew-"+str(nTors)+"-0.4",
            "skew-"+str(nTors)+"-0.5",
            "skew-"+str(nTors)+"-0.6",
            "skew-"+str(nTors)+"-0.7",
            "skew-"+str(nTors)+"-0.8",
            "skew-"+str(nTors)+"-0.9"]"""
# matrices = ["traffic_matrix_layer01_pass0001",
# "traffic_matrix_layer01_pass0002",
# "traffic_matrix_layer01_pass0003",
# "traffic_matrix_layer01_pass0004",
# "traffic_matrix_layer01_pass0005", ]

matrices = ["mixtral"]
#matrices = ["matrix"]
#%%
def decompose_into_perfect_matchings(Gx, N):
    # Ensure the graph is directed and multigraph
    assert isinstance(Gx, nx.MultiDiGraph), "Graph must be a directed multigraph."

    G = Gx    
    
    # Initialize the matchings
    matchings = [[] for _ in range(N)]
    numMatched = 0
    # For each node, process its outgoing and incoming edges separately
    edge_list = list(G.edges(keys=True))  # (u, v, key) tuples
    edge_keys = {}

    for i in range(N):
        for j in range(N):
            edge_keys[(i,j)] = list()

    for u, v, key in edge_list:
        edge_keys[(u,v)].append(key)

    nNodes = G.number_of_nodes()

    # Assign edges to matchings
    while G.degree(0) >=1:
        # print("DEGREE ", G.degree(0))
        GPrime = nx.Graph()
        for i in range(2*nNodes):
            if i < nNodes:
                GPrime.add_node(i, bipartite=0)
            else:
                GPrime.add_node(i, bipartite=1)
                
        
        for edge in G.edges():
            GPrime.add_edge(edge[0], edge[1]+nNodes)

        matching = nx.bipartite.maximum_matching(GPrime, top_nodes=[node for node in range(nNodes)])
        # print(matching)
        for u in matching:
            # print(u,matching[u])
            if u < nNodes:
                v = matching[u] - nNodes
                e=(u,v)
            else:
                break
            matchings[numMatched].append(e)
            # print(e)
            # keyp = edge_keys[e].pop(0)
            G.remove_edge(e[0], e[1],key=None)
        numMatched+=1

    return matchings


def trim_floats(val, tolerance=1e-9):
    if abs(val - round(val)) < tolerance:
        return round(val)
    return val

# def generalized_rounding(M, N, d):
#     model = gp.Model()
#     entry_vars = {}
#     row_sums = {}
#     col_sums = {}
#     row_sum_vars = {}
#     col_sum_vars = {}
#     #Every entry in demand matrix has integer var that is either floor or ceiling
#     for i in range(N):
#         for j in range(N):
#             if i !=j:
#                 entry_vars[i,j] = model.addVar(vtype=GRB.INTEGER,name=f"entry_{i}_{j}", lb=np.floor(M[i,j]), ub= np.ceil(M[i,j]))
#     for i in range(N):
#         row_sums[i] = trim_floats(np.sum(M[i,:])) #Necessary to trim sum floats, because otherwise you get 16.000000000004 which can be rounded up -> Not intended
#         col_sums[i] = trim_floats(np.sum(M[:,i]))
#         row_sum_vars[i] = model.addVar(vtype=GRB.INTEGER,name=f"row_sum_{i}", lb=np.floor(row_sums[i]), ub = np.ceil(row_sums[i]))
#         col_sum_vars[i] =  model.addVar(vtype=GRB.INTEGER,name=f"col_sum_{i}", lb=np.floor(col_sums[i]), ub = np.ceil(col_sums[i]))

#     #Add row and col constraints
#     for i in range(N):
#         model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == row_sum_vars[i]) # Row Sum
#         model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == col_sum_vars[i]) # Col Sum

#     model.update()
#     const = model.addVar(vtype=GRB.INTEGER, ub=0)#In this case, Vamsi mentioned that objective is unnecessary
#     model.setObjective(const, GRB.MAXIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
#     # Optimize the model
#     model.optimize()
#     #Create Rounded Matrix from gurobi variable values and return reference to that matrix
#     sol = np.zeros((N,N))
#     for i in range(N):
#         for j in range(N):
#             if i !=j:
#                 sol[i,j] = entry_vars[i,j].X
#     # print(np.array2string(sol)) #Debug rounding solution print statement
#     return sol

def return_normalized_matrix(M):
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    print(M)
    return M

def vermillion_graph(saturated_demand, k, N):
    normalized_demand = return_normalized_matrix(saturated_demand)
    deg = ((k-1)*N)
    scaled_demand = normalized_demand * deg
    print(scaled_demand)
    rounded_matrix = np.floor(scaled_demand) #generalized_rounding(scaled_demand, N, deg)
    out_Left  = []
    in_Left = []
    for row in range(N):
        out_Left.append(int(deg - np.sum(rounded_matrix[row,:])))
    for col in range(N):
        in_Left.append(int(deg - np.sum(rounded_matrix[:,col])))
    print(out_Left)
    print(in_Left)
    G = nx.directed_configuration_model(in_Left, out_Left)
    total_edge_cap = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                total_edge_cap[i,j] = rounded_matrix[i,j] + G.number_of_edges(i,j) +1

    # for i in range(N):
    #     val = total_edge_cap[i].sum()
    #     assert val == k * (N) + (N-1), f"Assertion failed for index {i}, total_edge_cap[i] = {val}"


    G = nx.MultiDiGraph()
    idx=0
    for i in range(N):
        for j in range(N):
            for _ in range(int(total_edge_cap[i,j])):
                G.add_edge(i,j,key=idx)
                idx+=1
    
    # nx.draw(G)
    return G, total_edge_cap

#%%

reconf = int(reconf*(10**6)) # in picoseconds

guard_band = int(guard_band*(10**6)) # in picoseconds

slottime = (reconf + guard_band)*9

for matrix in matrices:
    loaded_demand = np.loadtxt(matrix+".txt", usecols=range(N))
    np.fill_diagonal(loaded_demand, 0)
    eps = 1e-5
    loaded_demand[loaded_demand < eps] = 0
    filtered_demand = loaded_demand
    saturated_demand = filtered_demand * dE
    i = 0
    from pathlib import Path

    # Build directory path
    topo_dir = Path("./topologies")
    topo_dir.mkdir(parents=True, exist_ok=True)  

    # Build file path
    filename = (
        f"vermilion-{matrix}.txt"
    )
    out_path = topo_dir / filename

    # Write file
    with open(out_path, "w", encoding="utf-8") as outputfile:
        print(str(totalServers)+' '+str(nDownlinks)+' '+str(nUplinks)+' '+str(nTors), file=outputfile)
    
        period=int(k_s[i] * nTors/nUplinks)
        nSlice = 3*period

        print(str(nSlice)+' '+str(slottime)+' '+str(guard_band)+' '+str(reconf), file=outputfile)
        
        
        GVermilion, edge_cap = vermillion_graph(saturated_demand, k_s[i], N)
        G = copy.deepcopy(GVermilion)
        matchings = decompose_into_perfect_matchings(G, k_s[i]*N)
        
        adjacency={}
        adj = np.zeros((N,N))
        for timeslot in range(period):
            adjacency[timeslot] = np.zeros((nTors,nTors))
            for u in range(nTors):
                for v in range(nTors):
                    adjacency[timeslot][u][v]=0

        for timeslot in range(period):
            for index in range(dE):
                matching = timeslot*dE + index
                for edge in matchings[matching]:
                    adjacency[matching%period][edge[0]][edge[1]] += 1
                    adj[edge[0]][edge[1]] +=1

        print("conn={", file=outputfile)
        for slot in range(period):
            # Tx time interval
            print("{", file=outputfile)
            for u in range(nTors):
                printString = "{"
                for v in range(nTors-1):
                    printString+=str(int(adjacency[slot][u][v]))+','
                printString+=str(int(adjacency[slot][u][nTors-1]))
                if (u!=nTors-1):
                    printString+="},"
                else:
                    printString+="}"
                print(printString, file=outputfile)
            if (slot!=period-1):
                print("},", file=outputfile)
            else:
                print("}", file=outputfile)
            # print("\n",file=outputfile)
        print("};", file=outputfile)
#%%
