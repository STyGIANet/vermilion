#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:09:48 2024

@author: vamsi
"""
#%%

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np

#%%
# Create a random regular graph, just as an example
N = 8  # number of nodes
d = 4   # degree
G = nx.random_regular_graph(d, N)

# Alternatively, you could create your own graphs as follows.
# G = nx.DiGraph() # Directed graph object
# for i in range(N):
#     G.add_node(i) # add each node to the graph object
    
# # Depending on the topology you would like to construct, add edges as follows
# G.add_edge(1,2) # this adds an edge from 1 to 2 to the graph.

#%%

# Create a dictionary to store capacity variables for vertex pairs
capacity = {}
for i in range(N):
    for j in range(N):
        if i != j:
            if (i,j) in G.edges:
                capacity[(i,j)]=1
            else:
                capacity[(i,j)]=0

#%%

# Create All-to-All uniform demand matrix
demand = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j:
            demand[i][j] = d/(N-1)

#%%
##################################################################################
# Everything from here on, remains the same for any topology and demand instance
##################################################################################

# Generate all simple paths for each pair of nodes as a sequence of edges
all_paths = {}
for u in G.nodes():
    for v in G.nodes():
        if u!=v:
            all_paths[u, v] = [list(zip(path, path[1:])) for path in nx.all_simple_paths(G, source=u, target=v)]
            # Notice that nx.all_simple_paths will give each path as a sequence of nodes.
            # for path in nx.all_simple_paths(G, source=u, target=v):
            #     print(u,v,path)
            
            # We would like to have paths as sequence of edges.
            # all_paths[u, v] = [list(zip(path, path[1:])) for path in nx.all_simple_paths(G, source=u, target=v)]
            
            # Check how the paths are converted to sequence of edges
            # for path in all_paths[u,v]:
            #     print(u,v,path)

#%%

# Initialize the model
model = gp.Model("throughput")

#%%

# Add flow variables for each path
flow_vars = {}
for (s, d), paths in all_paths.items():
    for p in range(len(paths)):
        # set the variable type to continuous and set a lower bound of 0 (flow must not be negative)
        flow_vars[s, d, p] = model.addVar(vtype=GRB.CONTINUOUS, name=f"flow_{s}_{d}_{p}", lb=0)

#%% 
# Add Throughput variable and set the lower bound to 0 and upper bound to 1.
throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)

#%%

# Add constraints for edge capacities
for u, v in G.edges():
    model.addConstr(
        gp.quicksum(flow_vars[i, j, p] for (i, j), paths in all_paths.items() for p in range(len(paths)) if (u, v) in paths[p]) <= capacity[(u,v)],
        name=f"cap_{u}_{v}"
    )

#%%

# Add demand constraints

for u in G.nodes():
    for v in G.nodes():
        if u!=v:
            model.addConstr(
                gp.quicksum(flow_vars[u,v,p] for p in range(len(all_paths[u,v]))) >= throughput*demand[u][v],
                name=f"demand_{u}_{v}"
            )
#%%

# Set the objective to maximize throughput
model.setObjective(throughput, GRB.MAXIMIZE)

# Optimize the model
model.optimize()

print("Throughput for the given topology and the given demand matrix is:", throughput.X)