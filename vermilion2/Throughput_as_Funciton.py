import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matrixModification as mm
import Floor_Draft as fd
import Rounding_Draft as rd
import pandas as pd

#This file contains all kinds of useful functions; Most importantly calculating throughput for a given topology and M

def addGraphToMatrix(G, C): #Increments Matrix for edges of a RRG
    for i, j in G.edges:
        C[i, j] += 1
        C[j, i] += 1
def match_and_increment(list_a, list_b, C): #Graph Configuration Model; list_a is outgoing links left and list_b incoming links left
    
    
    N = len(list_a)
    
    # Continue until all entries in list_a are zero
    while any(val > 0 for val in list_a):
        # Randomly choose an index with a value > 0 from list_a
        a_index = random.choice([i for i in range(N) if list_a[i] > 0])
        
        # Randomly choose a different index with a value > 0 from list_b

        valid_b_indices = [i for i in range(N) if list_b[i] > 0 and i != a_index]
        if not valid_b_indices:
            break
        b_index = random.choice(valid_b_indices)
        # Decrement both link capacities at the selected indices
        list_a[a_index] -= 1
        list_b[b_index] -= 1
        
        C[a_index, b_index] +=1 #Increase capacity for constructed link


import random
def filtering(M, eps=1e-5):
    M[M < eps] = 0
    np.fill_diagonal(M, 0)
def return_normalized_matrix(M): #Normalizes a matrix by dividing it by the scalar which is the biggest row or col sum; Afterwards every row/col sum leq 1 
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    return M

def thetaEdgeFormulation(G, M, N, input_graph = True, measure_SH = False):#Given topology G and demand matrix M, returns throughput with optimal routing
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 0
    total_flow =0
    SH_flow = 0
    if input_graph:
        for i in range(N):
            for j in range(N):
                    capacity[(i,j)] = G.number_of_edges(i,j) 
                    if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]
    else:
        for i in range(N):
            for j in range(N):
                if i !=j:
                    capacity[(i, j)] = G[i,j]
    flow_variables = {}
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0:
                flow_variables[(i, j)] = {} 
                #Also if there is no demand between a pair we also don't need flow variables
                for s in range(N):
                    for d in range(N):
                        if s != d and M[s,d] > 0:
                            # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                            flow_variables[(i, j)][(s, d)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}_{s}_{d}', lb=0)
    throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
    model.update()
    # Implement the source demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                # Define the source demand constraint
                source_demand_constraint_expr = gp.quicksum(flow_variables[(s, i)][(s, d)] for i in range(N) if i != s and capacity[(s,i)] > 0) >= throughput * M[s, d]
                model.addConstr(source_demand_constraint_expr, f'source_demand_constraint_{s}_{d}')
    
    # Implement the destination demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                # Define the destination demand constraint
                dest_demand_constraint_expr = gp.quicksum(flow_variables[(i, d)][(s, d)] for i in range(N) if i != d and capacity[(i,d)] > 0) >= throughput * M[s, d]
                model.addConstr(dest_demand_constraint_expr, f'dest_demand_constraint_{s}_{d}')
    
    # Implement the flow conservation constraints for all j in N (excluding s and d), s in N, and d in N
     #Change order of loops
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                for j in range(N):
                    if j != s and j != d:  # Ensuring s, j, and d are all different
                        # Define the flow conservation constraint
                        flow_conservation_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for i in range(N) if i != j and capacity[(i, j)] > 0) - gp.quicksum(flow_variables[(j, k)][(s, d)] for k in range(N) if k != j and capacity[(j, k)] > 0) == 0
                        model.addConstr(flow_conservation_constraint_expr, f'flow_conservation_constraint_{j}_{s}_{d}')

    # Implement the capacity constraints for every i in N, for every j in N
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i,j)] >0:
                capacity_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for s in range(N) for d in range(N) if s != d and M[s,d] > 0) <= capacity[(i, j)]
                model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
    
    # Set the objective to maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)
    
    # Optimize the model
    model.optimize()
    # for v in model.getVars():
    #     if(v.x != 0):
    #         print(v.varName, "=", v.x)
    if(measure_SH):
        for s in range(N):
            for d in range(N):
                for i in range(N):
                    for j in range(N):
                        if i!=j and s!=d and M[s,d] != 0 and capacity[(i, j)] > 0:
                            total_flow+=flow_variables[(i, j)][(s,d)].X
                            if(i== s and d == j):
                                SH_flow+= flow_variables[(i, j)][(s,d)].X
        return (throughput.X,(total_flow, SH_flow))
    
    
    return throughput.X


def findavgRRGtheta(M, N, d, iter):#Find average throughput RRGs achieve over iter iterations
    thetas = []
    # SH = []
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta = thetaEdgeFormulation(G_temp, M, N, measure_SH=False)
        thetas.append(theta)
    return np.mean(thetas)
def createRingGraph(N, d):
    RingG= nx.MultiDiGraph()#d-Strong Ring
    for i in range(N):
        j = (i+1) % N
        for k in range(d): 
            RingG.add_edge(i,j)
    return RingG
if __name__ == "__main__":
    workdir="/tmp/"
