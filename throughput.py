#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:07:08 2023

@author: vamsi
"""

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import  random
import sys
from multiprocessing import Pool
import itertools

#%%

def createGraph(nNodes):
    G = nx.DiGraph()
    for v in range(nNodes):
        G.add_node(v)
    return G
    
def findpermutations(S,m):
    return set(itertools.permutations(S, m))

def findSubsets(S):
    ret = set()
    for i in range(len(S)):
        if i>4:
            continue
        # print(i)
        combinations = set(itertools.combinations(S, i))
        for j in combinations:
            ret.add(j)
    return ret

def generatePaths(G):
    paths={}

    for source in G.nodes:
        for sink in G.nodes:
            if source!=sink:
                paths[(source,sink)]=list()
                for path in nx.all_simple_edge_paths(G, source=source, target=sink):
                     paths[(source,sink)].append(path)
    return paths

def modelAddFlowVars(P,m):
    flow={}
    for i in P:
        for path in P[i]:
            flow[tuple(path)]=m.addVar(lb=0,vtype="C",name='flow'+str(path))
    return flow

# def modelAddTimeslotVars(schedules,m):
#     timeslots={}
#     for i in schedules:
#         timeslots[tuple(i)]=m.addVar(lb=0, ub=1, vtype="I",name='timeslot'+str(schedule))
#     return timeslots

#%%

nt = 4
nu = 1

switchPorts = np.arange(nt*nu)

matchings = list(findpermutations(switchPorts, len(switchPorts)))

schedules = list(findSubsets(matchings))

#%%

uniongraphs=list()
periods=list()
allPaths=list()
capacities=list()
count = 0
for i in range(len(schedules)):
    G = createGraph(nt)
    for matching in schedules[i]:
        for u in range(len(matching)):
            v = matching[u]
            try:
                G[u][v]['cap']+=1
            except:
                G.add_edge(u, v)
                G[u][v]['cap']=1
    # print('paths-'+str(count))
    paths={}
    for source in G.nodes:
        for sink in G.nodes:
            if source!=sink:
                paths[(source,sink)]=list()
                for path in nx.all_simple_edge_paths(G, source=source, target=sink):
                      paths[(source,sink)].append(path)
    allPaths.append(paths)
    periods.append(len(schedules[i]))
    uniongraphs.append(G)

    # if not (count%100):
    #     print('done-'+str(count))
    count+=1

#%%
C = 1
delta=0.1
demand=[ [0]*nt for i in range(nt)]

#### All to All
# for s in range(nt):
#     for d in range(nt):
#         if (s!=d):
#             demand[s][d] = C*nu/(nt-1)
#         else:
#             demand[s][d] = 0

### All to half
for s in range(nt):
    if s<nt-2:
        demand[s][s+1] = C*nu/2
        demand[s][s+2] = C*nu/2
    if s==nt-2:
        demand[s][nt-1] = C*nu/2
        demand[s][0] = C*nu/2
    if s==nt-1:
        demand[s][0] = C*nu/2
        demand[s][1] = C*nu/2

#### Permutation
# for s in range(nt):
#     for d in range(nt):
#         if (s==d+1) or (s==0 and d==nt-1):
#             demand[s][d] = C*nu
#         else:
#             demand[s][d] = 0

##### All to one
# for s in range(nt):
#     for d in [nt-1]:
#         if s!=d:
#             demand[s][d] = C*nu/(nt-1)
#         else:
#             demand[s][d] = 0            

##### One to All
# for d in range(nt):
#     for s in [nt-1]:
#         if s!=d:
#             demand[s][d] = C*nu/(nt-1)
#         else:
#             demand[s][d] = 0
            
            
####### Custom
# demand[0][0]= C
# demand[0][1]=C
# demand[0][2]=C
# demand[0][3]=C
# demand[1][0]= C
# demand[1][1]=C
# demand[1][2]=C
# demand[1][3]=C
# demand[2][0]= C
# demand[2][1]=C
# demand[2][2]=C
# demand[2][3]=C
# demand[3][0]= C
# demand[3][1]=C
# demand[3][2]=C
# demand[3][3]=C

#%%

def addCapacityConstr(G,C,m,flow,paths,index):
    for e in G.edges:
        if e[0]!=e[1]:
            if periods[index]>1:
                m.addConstr(gp.quicksum(flow[tuple(path)] for sd in paths for path in paths[sd] if e in path) <= C*(1-delta)/periods[index], name='capacity-'+str(e[0])+'-'+str(e[1])+'-'+str(index))
            else:
                m.addConstr(gp.quicksum(flow[tuple(path)] for sd in paths for path in paths[sd] if e in path) <= C/periods[index], name='capacity-'+str(e[0])+'-'+str(e[1])+'-'+str(index))


def addDemandConstr(G,C,m,flow,paths,index,throughput):
    for s in range(nt):
        for d in range(nt):
            if s!=d:
                m.addConstr(gp.quicksum(flow[tuple(path)] for path in paths[(s,d)]) >= throughput * demand[s][d], name='demand-'+str(s)+'-'+str(d)+'-'+str(index))
                

maxthroughput = 0
for i in range(len(uniongraphs)):
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
        # Create Gurobi Model
        # m = gp.Model("throughput")
        
            # Add flow variable for each path
            flows = modelAddFlowVars(allPaths[i],m)
            
            # Throughput variable
            throughput = m.addVar(lb=0,ub=1,vtype="C",name='throughput')
            
            # Add capacity constraints
            addCapacityConstr(uniongraphs[i],C,m,flows,allPaths[i],i)
            
            # Add demand constraints
            addDemandConstr(G, C, m, flows, allPaths[i], i, throughput)
            
            # Minimize the cycle length
            m.setObjective(throughput,GRB().MAXIMIZE)
            
            # Maximize the throughput
            
            
            m.optimize()
            if throughput.X > maxthroughput:
                maxthroughput = throughput.X
                maxIndex = i

print('\n##### Demand Matrix #####\n')
for s in range(nt):
    for d in range(nt):
        print('%.2f'%demand[s][d],end="\t\t")
    print('\n')
print('\n##### Throughput #####\n')
print('Throughput = '+str(maxthroughput))
print('\n#### Temporal Graph #####\n')
print("Period="+str(len(schedules[maxIndex])))
timeslot = 0
for i in schedules[maxIndex]:
    print(str('Timeslot='+str(timeslot)))
    index = 0
    for d in i:
        print(str(index)+'-->'+str(d))
        index+=1
    timeslot+=1
    
    
#%%