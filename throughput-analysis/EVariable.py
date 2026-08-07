import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from gurobipy import Model, GRB
import matrixModification as mm
import Throughput_as_Function as fct

#File used for calculating Optimal Throughput

def perfect_theta(N, d, M, measure_SH = False):#MIP that returns Optimal throughput or Opt throughput and tuple of total flow and single-hop flow.
    model = gp.Model("throughput")
    model.Params.LogToConsole = 0
    model.setParam('TimeLimit', 3600)#One hour at most
    #Add edge vars
    edge_vars = {}
    for i in range(N):
        for j in range(N):
            edge_vars[i, j] = model.addVar(vtype= GRB.INTEGER, name=f"edge_{i}_{j}",ub=d)
    #Inc edges constraint
    for i in range(N):
        model.addConstr(gp.quicksum(edge_vars[i,j] for j in range(N))<=d)
    #Outg edges constraint
    for i in range(N):
        model.addConstr(gp.quicksum(edge_vars[j,i] for j in range(N))<=d)

    #Add flow vars for every node combination
    flow_vars = {}
    for s in range(N):
        for d in range(N):
            if M[s,d] > 0:
                for i in range(N):
                    for j in range(N):
                        flow_vars[s, d, i, j] = model.addVar(vtype= GRB.CONTINUOUS, name=f"flow_{s}_{d}_{i}_{j}",lb=0)

    #Add throughput variable
    throughput = model.addVar(vtype=GRB.CONTINUOUS,name='throughput',lb=0,ub=1)
    
    #Source demand constraint
    for s in range(N):
        for d in range(N):
                if M[s,d] > 0:
                    model.addConstr(gp.quicksum(flow_vars[s, d, s, i] for i in range(N) if i!=s )>= M[s,d]* throughput,name =f"sdconst_{s}_{d}")

    #Dest demand constraint
    for s in range(N):
        for d in range(N):
                if M[s,d] > 0:
                    model.addConstr(gp.quicksum(flow_vars[s, d, i, d] for i in range(N) if i!=d )>= M[s,d]* throughput,name =f"ddconst_{s}_{d}")

    #Flow conservation
    for s in range(N):
        for d in range(N):
            if M[s,d] > 0:
                for j in (j for j in range(N) if j != s and j != d):
                    model.addConstr(gp.quicksum(flow_vars[s, d, i, j] for i in range(N) if i!=j )-gp.quicksum(flow_vars[s, d, j, k] for k in range(N) if k!=j)== 0,name =f"fcconst_{s}_{d}_{j}")

    #Capacity Constraints
    for i in range(N):
        for j in range(N):
            model.addConstr(gp.quicksum(flow_vars[s,d,i,j] for d in range(N) for s in range(N) if M[s,d] > 0)<= edge_vars[i,j],name =f"cconst_{s}_{d}")

    #Set the objective: maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)

    
    model.optimize()
    
    total_flow =0
    SH_flow = 0

    if(measure_SH):
        for s in range(N):
            for d in range(N):
                for i in range(N):
                    for j in range(N):
                        if i!=j and s!=d and M[s,d] > 0:
                            total_flow+=flow_vars[s,d,i,j].X
                            if(i== s and d == j):
                                SH_flow+= flow_vars[s,d,i,j].X
        return (model.objVal,(total_flow, SH_flow))

    return(model.objVal)
if(__name__ == "__main__"):
    matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(matrixdir+"skew-16-0.5"+".mat", usecols=range(16))
    demandMatrix = fct.prepareOrganicMatrix(demandMatrix)
    demandMatrix = demandMatrix *14
    print(perfect_theta(16, 14, demandMatrix))