import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import Rounding_Draft as rd
import Floor_Draft as fd
import matplotlib.pyplot as plt


#%%
workdir="/home/vamsi/src/phd/codebase/rdcn-throughput/matrices/"
matrices8 = [
    "chessboard-8",
    "uniform-8",
    "permutation-8",
    "skew-8-0.1",
    "skew-8-0.2",
    "skew-8-0.3",
    "skew-8-0.4",
    "skew-8-0.5",
    "skew-8-0.6",
    "skew-8-0.7",
    "skew-8-0.8",
    "skew-8-0.9",
]
matrices16 = [
            "chessboard-16",
            "uniform-16",
            "permutation-16",
            "skew-16-0.1",
            "skew-16-0.2",
            "skew-16-0.3",
            "skew-16-0.4",
            "skew-16-0.5",
            "skew-16-0.6",
            "skew-16-0.7",
            "skew-16-0.8",
            "skew-16-0.9",
            "data-parallelism","hybrid-parallelism","heatmap2","heatmap3", "topoopt"]
matrices32 = [
    "chessboard-32",
    "uniform-32",
    "permutation-32",
    "skew-32-0.1",
    "skew-32-0.2",
    "skew-32-0.3",
    "skew-32-0.4",
    "skew-32-0.5",
    "skew-32-0.6",
    "skew-32-0.7",
    "skew-32-0.8",
    "skew-32-0.9",
]
matrices48 = [
    "chessboard-48",
    "uniform-48",
    "permutation-48",
    "skew-48-0.1",
    "skew-48-0.2",
    "skew-48-0.3",
    "skew-48-0.4",
    "skew-48-0.5",
    "skew-48-0.6",
    "skew-48-0.7",
    "skew-48-0.8",
    "skew-48-0.9",
]
matrices64 = [
    "chessboard-64",
    "uniform-64",
    "permutation-64",
    "skew-64-0.1",
    "skew-64-0.2",
    "skew-64-0.3",
    "skew-64-0.4",
    "skew-64-0.5",
    "skew-64-0.6",
    "skew-64-0.7",
    "skew-64-0.8",
    "skew-64-0.9",
]
matrices128 = [
    "chessboard-128",
    "uniform-128",
    "permutation-128",
    "skew-128-0.1",
    "skew-128-0.2",
    "skew-128-0.3",
    "skew-128-0.4",
    "skew-128-0.5",
    "skew-128-0.6",
    "skew-128-0.7",
    "skew-128-0.8",
    "skew-128-0.9",
]
matrices256 = [
    "chessboard-256",
    "uniform-256",
    "permutation-256",
    "skew-256-0.1",
    "skew-256-0.2",
    "skew-256-0.3",
    "skew-256-0.4",
    "skew-256-0.5",
    "skew-256-0.6",
    "skew-256-0.7",
    "skew-256-0.8",
    "skew-256-0.9",
]
matrices512 = [
    "chessboard-512",
    "uniform-512",
    "permutation-512",
    "skew-512-0.1",
    "skew-512-0.2",
    "skew-512-0.3",
    "skew-512-0.4",
    "skew-512-0.5",
    "skew-512-0.6",
    "skew-512-0.7",
    "skew-512-0.8",
    "skew-512-0.9",
]
matrices1024 = [
    "chessboard-1024",
    "uniform-1024",
    "permutation-1024",
    "skew-1024-0.1",
    "skew-1024-0.2",
    "skew-1024-0.3",
    "skew-1024-0.4",
    "skew-1024-0.5",
    "skew-1024-0.6",
    "skew-1024-0.7",
    "skew-1024-0.8",
    "skew-1024-0.9",
]

def trim_floats(val, tolerance=1e-9):
    if abs(val - round(val)) < tolerance:
        return round(val)
    return val
def thetaSingleHop(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable with single hops only
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 1
    # model.Params.OptimalityTol = 1e-9
    # model.Params.FeasibilityTol = 1e-9
    # model.Params.IntFeasTol = 1e-9
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
            if i != j and capacity[(i, j)] > 0 and M[i,j] > 0:
                # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                flow_variables[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}', lb=0)
    throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
    model.update()
    # Implement throughput constraint
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0 and M[i,j] > 0:
                # Define the throughput constraint
                throughput_constraint_expr = flow_variables[(i,j)] >= throughput * M[i, j]
                model.addConstr(throughput_constraint_expr, f'throughput_constraint_{i}_{j}')
                # Define the capacity constraint
                capacity_constraint_expr = flow_variables[(i, j)] <= capacity[(i, j)]
                model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
    # Set the objective to maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)
    # Optimize the model
    model.optimize()
    return throughput.X

def generalized_rounding(M, N, d):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
    model = gp.Model()
    entry_vars = {}
    row_sums = {}
    col_sums = {}
    row_sum_vars = {}
    col_sum_vars = {}
    #Every entry in demand matrix has integer var that is either floor or ceiling
    for i in range(N):
        for j in range(N):
            if i !=j:
                entry_vars[i,j] = model.addVar(vtype=GRB.INTEGER,name=f"entry_{i}_{j}", lb=np.floor(M[i,j]), ub= np.ceil(M[i,j]))
    for i in range(N):
        row_sums[i] = trim_floats(np.sum(M[i,:])) #Necessary to trim sum floats, because otherwise you get 16.000000000004 which can be rounded up -> Not intended
        col_sums[i] = trim_floats(np.sum(M[:,i]))
        row_sum_vars[i] = model.addVar(vtype=GRB.INTEGER,name=f"row_sum_{i}", lb=np.floor(row_sums[i]), ub = np.ceil(row_sums[i]))
        col_sum_vars[i] =  model.addVar(vtype=GRB.INTEGER,name=f"col_sum_{i}", lb=np.floor(col_sums[i]), ub = np.ceil(col_sums[i]))

    #Add row and col constraints
    for i in range(N):
        model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == row_sum_vars[i]) # Row Sum
        model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == col_sum_vars[i]) # Col Sum

    model.update()
    const = model.addVar(vtype=GRB.INTEGER, ub=0)#In this case, Vamsi mentioned that objective is unnecessary
    model.setObjective(const, GRB.MAXIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix from gurobi variable values and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    # print(np.array2string(sol)) #Debug rounding solution print statement
    return sol
def return_normalized_matrix(M): #Normalizes a matrix by dividing it by the scalar which is the biggest row or col sum; Afterwards every row/col sum leq 1 
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    return M # call-by-reference doesn't work for some reason, hence returning M
def vermillion_throughput(saturated_demand, d, k, N, MH = True):
    normalized_demand = return_normalized_matrix(saturated_demand)
    deg = ((k-1)*N)
    scaled_demand = normalized_demand * deg
    rounded_matrix = generalized_rounding(scaled_demand, N, deg)
    out_Left  = []
    in_Left = []
    for row in range(N):
        out_Left.append(int(deg - np.sum(rounded_matrix[row,:])))
    for col in range(N):
        in_Left.append(int(deg - np.sum(rounded_matrix[:,col])))
    print(out_Left)
    print(in_Left)
    G = nx.directed_configuration_model(in_Left, out_Left) #TODO: Re-Check GCM in normal Floor heuristic: Need to use list; not dict!
    # nx.draw_circular(G, with_labels= True)
    # plt.show()
    # for i in range(N): #Debug print statements
    #     print("Node " + str(i) + " has the following edges:")
    #     for j in range(N):
    #         if i !=j and G.number_of_edges(i,j) != 0:
    #             print("To " + str(j) +" : "  + str(G.number_of_edges(i,j)))
    total_edge_cap = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                total_edge_cap[i,j] = rounded_matrix[i,j] + G.number_of_edges(i,j) +1
    total_edge_cap = total_edge_cap *(d /(k*N))
    # print(np.array2string(total_edge_cap)) #Debug capacity print statement
    
    thetaSH = thetaSingleHop(total_edge_cap, saturated_demand, N, input_graph=False)
    if(MH):
        return (fct.thetaEdgeFormulation(total_edge_cap,saturated_demand, N, input_graph=False ), thetaSH)
    else:
        return thetaSH
def rotornet_throughput(saturated_demand, d ,N):
    total_edge_cap = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            total_edge_cap[i,j] = d/N
            
    # return (fct.thetaPathFormulation(total_edge_cap, saturated_demand,N, False), thetaSingleHop(total_edge_cap, saturated_demand,N, False))
    return (fct.thetaEdgeFormulation(total_edge_cap, saturated_demand,N, False), thetaSingleHop(total_edge_cap, saturated_demand,N, False))

#%%


#%%

if __name__ == "__main__":
    NValues=[8,16,32,48,64,128,256,512,1024]
    dE = 4
    k_s=[2,3,4,5,6]
    matrices=[matrices8, matrices16, matrices32, matrices48, matrices64, matrices128, matrices256, matrices512, matrices1024]
    with open("nsdi-throughput-results.csv", "w") as outputfile:
        print("#######################################################")
        print("mygrep","n","d","matrix","alg","k","throughputSH","throughputMH")
        print("#######################################################")
        print("mygrep","n","d","matrix","alg","k","throughputSH","throughputMH",file=outputfile)
        for i in range(len(NValues)):
            N = NValues[i]
            matrixSet = matrices[i]
            for matrix in matrixSet:
                loaded_demand = np.loadtxt(workdir+matrix + ".mat", usecols=range(N))
                eps = 1e-5
                loaded_demand[loaded_demand < eps] = 0 # Filter loaded demand?
                filtered_demand = return_normalized_matrix(loaded_demand)
                saturated_demand = filtered_demand * dE
                # print(np.array2string(saturated_demand))
                if N <= 48:
                    rotor_res = rotornet_throughput(saturated_demand,dE, N)
                    print("#######################################################")
                    print("mygrep",N,dE, matrix, "rotornet",1,rotor_res[1],rotor_res[0])
                    print("#######################################################")
                    print("mygrep",N,dE, matrix, "rotornet",1,rotor_res[1],rotor_res[0],file=outputfile)
                
                for j in range(len(k_s)):
                    vermilion_res = vermillion_throughput(saturated_demand, dE, k_s[j], N, MH = False)
                    print("#######################################################")
                    print("mygrep",N,dE, matrix, "vermilion",k_s[j],vermilion_res,1)
                    print("#######################################################")
                    print("mygrep",N,dE, matrix, "vermilion",k_s[j],vermilion_res,1,file=outputfile)
    
    
    # N = 16
    # k_s = [2,3,4,5]
    # rotornet_res = []
    # Vermillion_res = []
    # for matrix in matrices16:
    #     loaded_demand = np.loadtxt(workdir+matrix + ".mat", usecols=range(N))
    #     # print(np.array2string(loaded_demand))
    #     eps = 1e-5
    #     # print("_________________________________________________")
    #     loaded_demand[loaded_demand < eps] = 0 # Filter loaded demand?
    #     # print(np.array2string(loaded_demand))

    #     filtered_demand = loaded_demand
    #     saturated_demand = filtered_demand * dE
    #     # print(np.array2string(saturated_demand))
    #     rotornet_res.append(rotornet_throughput(saturated_demand,dE, N))
    #     k_results = []
    #     for i in range(len(k_s)):
    #         k_results.append(vermillion_throughput(saturated_demand, dE, k_s[i], N, MH = False))
    #     Vermillion_res.append(k_results)
    # print("Results for each matrix using filtered Demand and Vermillion with k = " + str(k)) 
    # print("Results for each matrix using RotorNet and filtering") 
    # for i in range(len(matrices16)):
    #     result_string = ""
    #     result_string +=matrices16[i]
    #     for j in range(len(k_s)):
    #         result_string +="  |k=" +str(k_s[j])  + "  Verm SH: " +  str(Vermillion_res[i][j])
    #     print(result_string)
        
    # print(np.array2string(normalized_demand))
    # print(normalized_demand.sum(axis=0).max())  #Max Column Sum  
    # print(normalized_demand.sum(axis=1).max())  # Max Row Sum
    # print(max(normalized_demand.sum(axis=0).max(),normalized_demand.sum(axis=1).max()))
    # print(np.array2string(arr))
    # print(arr.sum(axis=0).max())  #Max Column Sum  
    # print(arr.sum(axis=1).max())  # Max Row Sum
    # print(max(arr.sum(axis=0).max(),arr.sum(axis=1).max()))
    
    

    # print(np.array2string(np.floor(arr)))
    # print(max(arr.sum(axis=0).max(),arr.sum(axis=1).max()))
    
    #You need to adapt rounding, st. it corresponds to original row/col sums, not (k-1)*n, 
    
    