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
N = int(sys.argv[1])
# N=16

#%% 
# Set the degree for each vertex
degree = sys.argv[2]  # Define the degree variable
# degree=4
#%%
# List of matrices
workdir="/home/vamsi/src/phd/codebase/rdcn-throughput/matrices/"
matrices16 = [
            "chessboard-16",
            "uniform-16",
            "permutation-16",
            "skew-16-0.0",
            "skew-16-0.1",
            "skew-16-0.2",
            "skew-16-0.3",
            "skew-16-0.4",
            "skew-16-0.5",
            "skew-16-0.6",
            "skew-16-0.7",
            "skew-16-0.8",
            "skew-16-0.9",
            "skew-16-1.0",
            "data-parallelism","hybrid-parallelism","heatmap2","heatmap3"]

matrices64 = ["chessboard-64",
              "uniform-64",
              "permutation-64",
              "skew-64-0.0",
              "skew-64-0.1",
              "skew-64-0.2",
              "skew-64-0.3",
              "skew-64-0.4",
              "skew-64-0.5",
              "skew-64-0.6",
              "skew-64-0.7",
              "skew-64-0.8",
              "skew-64-0.9",
              "skew-64-1.0",]
degrees = [16, 14, 12, 10, 8, 6, 4]
matrices=matrices16
if N==16:
    matrices=matrices16
    degrees = [16, 14, 12, 10, 8, 6, 4]
elif N==64:
    matrices=matrices64
    degrees = [4, 8, 16, 32, 64]
else:
    print("Set N=16 or N=64")
    exit

#%%
print("mygrep,N,matrix,maxValue,networkType,degree,throughput")

degree = 16
matrixfile = "permutation-16"
for degree in degrees:
    for matrixfile in matrices:
        demand = np.loadtxt(workdir+matrixfile+".mat", usecols=range(N))
        demand=demand*degree # The original matrix is normalized. We scale the matrix to for the specified degree
        maxValue = np.ceil(np.max(demand))
        print("################")
        print(matrixfile, "static")
        print("################")
        
        Gstatic = nx.random_regular_graph(np.min([N-1,degree]), N)
        
        
        # Create a new Gurobi model
        model = gp.Model()
        
        # Create a dictionary to store capacity variables for vertex pairs
        capacity = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    if (i,j) in Gstatic.edges:
                        capacity[(i,j)]=1
                    else:
                        capacity[(i,j)]=0
        
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
        throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
        
        # Update the model to include these variables
        model.update()
        
        # Implement the source demand constraints for all s in N and all d in N
        for s in range(N):
            for d in range(N):
                if s != d:
                    # Define the source demand constraint
                    source_demand_constraint_expr = gp.quicksum(flow_variables[(s, i)][(s, d)] for i in range(N) if i != s) >= throughput * demand[s, d]
                    model.addConstr(source_demand_constraint_expr, f'source_demand_constraint_{s}_{d}')
        
        # Implement the destination demand constraints for all s in N and all d in N
        for s in range(N):
            for d in range(N):
                if s != d:
                    # Define the destination demand constraint
                    dest_demand_constraint_expr = gp.quicksum(flow_variables[(i, d)][(s, d)] for i in range(N) if i != d) >= throughput * demand[s, d]
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
                    capacity_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for s in range(N) for d in range(N) if s != d) <= capacity[(i, j)]
                    model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
        
        # Set the objective to maximize throughput
        model.setObjective(throughput, GRB.MAXIMIZE)
        
        # Optimize the model
        model.optimize()
        
        # Check the optimization status
        if model.status == GRB.OPTIMAL:
            print("mygrep"+str(",")+str(N)+str(",")+str(matrixfile)+str(",")+str(maxValue)+str(",")+str("static")+str(",")+str(degree)+str(",")+f"{model.objVal}")
        else:
            print("mygrep"+str(",")+str(N)+str(",")+str(matrixfile)+str(",")+str(maxValue)+str(",")+str("static")+str(",")+str(degree)+str(",")+"NULL")
        
        # # Print capacity values (commented out)
        # '''
        # print("Capacity Values:")
        # for i in range(N):
        #     for j in range(N):
        #         if i != j:
        #             print(f'Capacity_{i}_{j} = {capacity[(i, j)].x}')
        # '''
        
        # You can also access the variable values if needed, e.g., flow_values = model.getAttr('x', flow_variables)
#%%