import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import matplotlib.pyplot as plt
import matrixModification as MM
import random
import sys
#%%
def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    ]
    for i in range(3):
        res[i] += str(N)
    for j in range(4):
        res[j+3] += str(N) + "-0." + str(2*(j+1))
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap1"]
matrixdir="./matrices/"
#%%
def trim_floats(val, tolerance=1e-9):
    if abs(val - round(val)) < tolerance:
        return round(val)
    return val
def thetaSingleHop(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable with single hops only
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 0
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
                if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]

    
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

def thetaSingleHop2(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable with single hops only
    capacity = np.zeros((N,N))
    if(input_graph):
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
                if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]
    
    # To avoid division by zero, handle elements where demand is zero
    # We set the ratio as infinity where demand is zero (so we ignore those cases).
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(M != 0, capacity / M, np.inf)
    
    # Find the smallest ratio (ignoring infinities)
    smallest = min(1, np.min(ratios))
    
    return smallest

def return_normalized_matrix(M): #Normalizes a matrix by dividing it by the scalar which is the biggest row or col sum; Afterwards every row/col sum leq 1 
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    return M # call-by-reference doesn't work for some reason, hence returning M
def randomized_vermillion_throughput(saturated_demand, saturated_noise, d, k, N, MH = True):
    normalized_demand = return_normalized_matrix(saturated_noise)
    deg = ((k-1)*N)
    scaled_demand = normalized_demand * deg
    

    floor_matrix = np.floor(scaled_demand)
    out_Left  = []
    in_Left = []
    for row in range(N):
        out_Left.append(int(deg - np.sum(floor_matrix[row,:])))
    for col in range(N):
        in_Left.append(int(deg - np.sum(floor_matrix[:,col])))
    # print(out_Left)
    # print(in_Left)


    total_edge_cap =floor_matrix

    fct.match_and_increment(out_Left, in_Left, total_edge_cap)



    # nx.draw_circular(G, with_labels= True)
    # plt.show()
    # for i in range(N): #Debug print statements
    #     print("Node " + str(i) + " has the following edges:")
    #     for j in range(N):
    #         if i !=j and G.number_of_edges(i,j) != 0:
    #             print("To " + str(j) +" : "  + str(G.number_of_edges(i,j)))
    
    for i in range(N):
        for j in range(N):
            if i !=j:
                total_edge_cap[i,j] += 1
    total_edge_cap = total_edge_cap *(d /(k*N))
    # print(np.array2string(total_edge_cap)) #Debug capacity print statement
    
    SH_res = thetaSingleHop2(total_edge_cap, saturated_demand, N, input_graph=False)
    if(MH):
        return (fct.thetaEdgeFormulation(total_edge_cap,saturated_demand, N, input_graph=False ), SH_res)
    else:
        return SH_res


#%%

if __name__ == "__main__":
    
    NValues=[8,16,32]
    # NValues=[8]
    dE = 4
    k_s=[1,2,3,4,5,6,7,8,9,10]
    noise_values = np.linspace(0, dE/3,10)
    
    N = 16
    noise = int(sys.argv[1])
    with open("results/sigmetrics-throughput-results-"+str(N)+"-"+str(noise)+".csv", "w") as outputfile:
        # print("matrix", "alg", "k", "N", "dE", "noise", "multadd", "throughput", file=outputfile)

        matrices=generate_synthmatrix_names(N)
        
        if N==16:
            matrices = matrices + organicmatrices16
        
        for matrix in matrices:
            loaded_demand = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
            # loaded_demand = loaded_demand * dE
            eps = 1e-5
            loaded_demand[loaded_demand < eps] = 0 # Filter loaded demand?
            filtered_demand = return_normalized_matrix(loaded_demand)
            
            saturated_noise = return_normalized_matrix(MM.add_additive_noise(filtered_demand, N, noise_values[noise])) * dE
            np.fill_diagonal(saturated_noise, 0)
            saturated_demand = filtered_demand * dE

            for k in k_s:
                res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, k , N)
                print(matrix, "vermThroughput", k, N, dE, noise, "add", res[0])
                print(matrix, "vermThroughput", k, N, dE, noise, "add", res[0] ,file=outputfile)

                
                randT = list()
                randR = list()
                for i in range(10):
                    res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, random.randint(1,k+1) , N)
                    randT.append(res[0])
                
                print(matrix, "randThroughput", k, N, dE, noise, "add", np.mean(randT))
                print(matrix, "randThroughput", k, N, dE, noise, "add", np.mean(randT) ,file=outputfile)

                
                
            saturated_noise = return_normalized_matrix(MM.add_multiplicative_noise(filtered_demand, N, noise_values[noise])) * dE
            np.fill_diagonal(saturated_noise, 0)
            saturated_demand = filtered_demand * dE

            for k in k_s:
                res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, k , N)
                print(matrix, "vermThroughput", k, N, dE, noise, "mult", res[0])
                print(matrix, "vermThroughput", k, N, dE, noise, "mult", res[0] ,file=outputfile)

                
                randT = list()
                randR = list()
                for i in range(10):
                    res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, random.randint(1,k+1) , N)
                    randT.append(res[0])
                
                print(matrix, "randThroughput", k, N, dE, noise, "mult", np.mean(randT))
                print(matrix, "randThroughput", k, N, dE, noise, "mult", np.mean(randT) ,file=outputfile)