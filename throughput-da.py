#%% 
# Import Statements
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import networkx as nx
import random
#%% 
# Set the number of vertices
# N = sys.argv[1]
N=12
# Set the degree for each vertex
# degree = sys.argv[2]  # Define the degree variable
degree=int(N/2)

#%%

demand = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j:
            demand[i][j]=1/3*(N/(N-1))

#%%
# Create a new Gurobi model
model = gp.Model()
G = nx.random_regular_graph(degree, N)

nx.draw(G,arrowstyle="->", arrows=True,label=True)
print(nx.diameter(G))


# Create a dictionary to store capacity variables for vertex pairs
capacity = {}
for i in range(N):
    for j in range(N):
        if i != j:
            if (i,j) in G.edges():
                capacity[(i,j)]=1
            else:
                capacity[(i,j)]=0
            # capacity[(i, j)] = model.addVar(vtype=GRB.INTEGER, name=f'capacity_{i}_{j}', lb=0)  # Set lb=0 for capacity

# Create a dictionary to store flow variables for each node pair (i, j) corresponding to all node pairs (s, d)
flow_variables = {}
for i in range(N):
    for j in range(N):
        if i != j:
            flow_variables[(i, j)] = {}
            for s in range(N):
                for d in range(N):
                    if s != d:
                        # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                        flow_variables[(i, j)][(s, d)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}_{s}_{d}', lb=0)

# Create a variable for throughput with lower bound 0
throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=100)

# Update the model to include these variables
model.update()

# Implement the source demand constraints for all s in N and all d in N
for s in range(N):
    for d in range(N):
        if s != d:
            # Define the source demand constraint
            source_demand_constraint_expr = gp.quicksum(flow_variables[(s, i)][(s, d)] for i in range(N) if i != s) >= 0.01*throughput * demand[s, d]
            model.addConstr(source_demand_constraint_expr, f'source_demand_constraint_{s}_{d}')

# Implement the destination demand constraints for all s in N and all d in N
for s in range(N):
    for d in range(N):
        if s != d:
            # Define the destination demand constraint
            dest_demand_constraint_expr = gp.quicksum(flow_variables[(i, d)][(s, d)] for i in range(N) if i != d) >= 0.01*throughput * demand[s, d]
            model.addConstr(dest_demand_constraint_expr, f'dest_demand_constraint_{s}_{d}')

# Implement the flow conservation constraints for all j in N (excluding s and d), s in N, and d in N
for j in range(N):
    for s in range(N):
        for d in range(N):
            if j != s and j != d and s != d:  # Ensuring s, j, and d are all different
                # Define the flow conservation constraint
                flow_conservation_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for i in range(N) if i != j) - gp.quicksum(flow_variables[(j, k)][(s, d)] for k in range(N) if k != j) == 0
                model.addConstr(flow_conservation_constraint_expr, f'flow_conservation_constraint_{j}_{s}_{d}')

# Implement the capacity constraints for every i in N, for every j in N
for i in range(N):
    for j in range(N):
        if i != j:
            # Define the capacity constraint
            capacity_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for s in range(N) for d in range(N) if s != d) <= capacity[(i, j)]
            model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')

# Implement the demand-aware outgoing links constraints for all s in N
for s in range(N):
    # Define the demand-aware outgoing links constraint
    outgoing_links_constraint_expr = gp.quicksum(capacity[(s, i)] for i in range(N) if i != s) - degree <= 0
    model.addConstr(outgoing_links_constraint_expr, f'outgoing_links_constraint_{s}')

# Implement the demand-aware incoming links constraints for all d in N
for d in range(N):
    # Define the demand-aware incoming links constraint
    incoming_links_constraint_expr = gp.quicksum(capacity[(i, d)] for i in range(N) if i != d) - degree <= 0
    model.addConstr(incoming_links_constraint_expr, f'incoming_links_constraint_{d}')

obj = model.addVar(vtype=GRB.CONTINUOUS, name='obj', lb=0)
mlu = model.addVar(vtype=GRB.CONTINUOUS, name='mlu', lb=0)

# Update the model to include these variables
model.update()

model.addConstr(mlu == gp.max_([capacity[(i,j)] for i in range(N) for j in range(N) if i!=j]))



# model.addConstr(throughput - obj + fsr == mlu )
model.addConstr(throughput - obj == mlu )

# Set the objective to maximize throughput
model.setObjective(obj, GRB.MAXIMIZE) 

# Optimize the model
model.optimize()
print("Throughput: ", throughput.X)
#%%

loads = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i!=j:
            loads[i][j] = np.sum([flow_variables[(i,j)][(s,d)].X for s in range(N) for d in range(N) if s!=d])
            
#%%

edgeset = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i!=j:
            edgeset[i][j] = capacity[(i,j)]
#%%

s = 1
d = 7
sumD=0
for s in range(N):
    for d in range(N):
        if s!=d:
            for i in range(N):
                for j in range(N):
                    if i!=j and flow_variables[(i,j)][(s,d)].X !=0:
                        print("Demand: ",s,"--->",d,"edge", i,"--->",j,"flow: ", flow_variables[(i,j)][(s,d)].X)
                        sumD = sumD + flow_variables[(i,j)][(s,d)].X
                        
                        
#%%

demand = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j:
            demand[i][j]=1/3*(N/(N-1))

for i in range (N):
    demand[i][i] = N/2 - np.sum(demand[i])


#%%
import networkx as nx
import math
from collections import defaultdict

# Utility function to compute the shortest path using Dijkstra's algorithm
def dijkstra_paths_lengths(graph, lengths, source):
    # Update the graph with the current edge lengths
    for (u, v) in graph.edges():
        graph[u][v]['weight'] = lengths.get((u, v), float('inf'))
    
    # Compute shortest paths from the source to all other nodes
    paths = nx.single_source_dijkstra_path(graph, source, weight='weight')
    distances = nx.single_source_dijkstra_path_length(graph, source, weight='weight')
    
    return paths, distances

# Main function to solve the Maximum Concurrent Multicommodity Flow Problem
def max_concurrent_multicommodity_flow(graph, capacities, demands_matrix, epsilon):
    # Calculate the number of edges
    m = graph.number_of_edges()
    
    # Set delta using the given formula
    delta = (1 / (1 + epsilon) ** ((1 - epsilon) / epsilon)) * ((1 - epsilon) / m) ** (1 / epsilon)
    
    # Initialize parameters
    lengths = {(u, v): delta / capacities.get((u, v), 1) for (u, v) in capacities}
    x = defaultdict(float)
    
    def D(lengths):
        return sum(capacities.get((u, v), 0) * lengths.get((u, v), 0) for (u, v) in capacities)
    
    while D(lengths) < 1:
        for i in range(graph.number_of_nodes()):
            sources_demands = [(i, j) for j in range(graph.number_of_nodes()) if demands_matrix[i][j] > 0]
            
            d0 = {(i, j): demands_matrix[i][j] for (i, j) in sources_demands}
            while D(lengths) < 1 and any(d0[(i, j)] > 0 for (i, j) in d0):
                paths, distances = dijkstra_paths_lengths(graph, lengths, i)
                
                P = {(i, j): paths.get(j, []) for (i, j) in d0 if j in paths and d0[(i, j)] > 0}
                
                # Debug information
                # print("Paths:", P)
                
                f = {}
                for (i, j) in P:
                    path_edges = [(P[(i, j)][k], P[(i, j)][k + 1]) for k in range(len(P[(i, j)]) - 1)]
                    if path_edges:
                        # Debug information
                        # print("Path edges for", (i, j), ":", path_edges)
                        # Compute the maximum flow scaling factor sigma
                        sigma = max(1, max(
                            (d0[(i, j)] / capacities.get(edge, float('inf')))
                            for edge in path_edges
                        ))
                        f[(i, j)] = d0[(i, j)] / sigma

                        # Debug information
                        # print("Sigma:", sigma)
                        
                        d0[(i, j)] -= f[(i, j)]
                        for edge in path_edges:
                            # Avoid KeyError by using get() with a default value
                            capacities_edge = capacities.get(edge, 1)
                            lengths[edge] = lengths.get(edge, 1) * (1 + epsilon * f[(i, j)] / capacities_edge)
                            x[edge] += f[(i, j)]
    
    # Final scaling of flow
    log_factor = math.log(1 + epsilon)
    for path in x:
        x[path] /= log_factor
    
    # Compute lambda using the final paths and flow values
    valid_paths = [
        sum(x.get((P[i], P[i + 1]), 0) for i in range(len(P) - 1)) / demands_matrix[P[0]][P[-1]]
        for P in P.values() if demands_matrix[P[0]][P[-1]] > 0 and len(P) > 1
    ]
    
    if not valid_paths:  # Handle the case where no valid paths exist
        lambda_val = 0
    else:
        lambda_val = min(valid_paths)
    
    return x, lambda_val



capacities = {(u, v): 1 for u, v in G.edges()}

# Demands matrix
demands_matrix = demand

epsilon = 0.01

# Run the algorithm
x, lambda_val = max_concurrent_multicommodity_flow(G, capacities, demands_matrix, epsilon)
print("Flow x:", dict(x))
print("Lambda value:", lambda_val)
