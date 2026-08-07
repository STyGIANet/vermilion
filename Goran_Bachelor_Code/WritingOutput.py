import numpy as np
import Throughput_as_Function as fct
import EVariable as ev
import matrixModification as mm

#This file was used to generate the results for all algorithms and print them out in the console and a text file

def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "random-skewed-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    ]
    for i in range(4):
        res[i] += str(N)
    for j in range(9):
        res[j+4] += str(N) + "-0." + str(j+1)
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3", "topoopt"] #Names of the DNN workloads
matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
outputdir = "/home/studium/Documents/Code/rdcn-throughput/Goran_Bachelor_Code/"
n_values_used = [8,16]
d_s = [2,3,4,5,6,7]
def oneIteration(N, M, matrix, outputfile):
    if(matrix in organicmatrices16):
        demandMatrix = fct.prepareOrganicMatrix(demandMatrix)
        matrix = "Sinkhorn_" + matrix
    for d in d_s:
        d = round(d*(N/8))
        saturatedM = M * d


        string_Beginning = str(N) +" "+ matrix +" " + str(d)


        RRG_theta =  fct.findavgRRGtheta(saturatedM, N, d, 10)
        RRG_string = string_Beginning + " RRG " + " " +str(RRG_theta)
        print(RRG_string)
        outputfile.write(RRG_string+"\n")

        Floor_theta = fct.findBestGamma(N, d, saturatedM)
        Floor_String =  f"{string_Beginning} Floor {Floor_theta:.5f}"
        print(Floor_String)
        outputfile.write( Floor_String+"\n")

        Rounding_theta =  fct.findBestGamma(N, d, saturatedM, Rounding=True)
        Rounding_String =f"{string_Beginning} Rounding {Rounding_theta:.5f}"
        print(Rounding_String)
        outputfile.write( Rounding_String+"\n")

        circle_theta = fct.thetaEdgeFormulation(fct.createRingGraph(N,d),saturatedM, N)
        circleString = string_Beginning + " Circle " +  str(circle_theta)
        print(circleString)
        outputfile.write( circleString+"\n")


def writePerfectTheta(N, matrices, d_lists):#For a given list of matrices and corresponding lists of values for d, print Optimal throughput and SH share
    for i in range(len(matrices)):
        matrix = matrices[i]
        demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
        demandMatrix = fct.return_normalized_matrix(demandMatrix)
        if(matrix) in organicmatrices16:
            demandMatrix = fct.prepareOrganicMatrix(demandMatrix)
            matrix = "Sinkhorn_" + matrix
        for d in d_lists[i]:
            saturatedM = demandMatrix * d
            string_Beginning = str(N) +" "+matrix +" " + str(d)

            res= ev.perfect_theta(N,d,saturatedM, measure_SH=True)
            perfect_theta = res[0]
            SH_Share = res[1][1] / res[1][0]
            perfect_string = string_Beginning + " Optimal " + str(SH_Share) + " " + str(perfect_theta)
            print(perfect_string)

if __name__ == "__main__":
    # all_ds = [4,6,8,10,12,14]
    # matrices = ["skew-16-0.5", "skew-16-0.7", "skew-16-0.3"] 
    # d_lists = [all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds,all_ds, all_ds, all_ds, all_ds]

    # writePerfectTheta(16, matrices, d_lists)




    print("N matrix d Alg throughput\n")
    outputfile = open(outputdir+"TestOutput", "w")
    outputfile.write("N matrix d Alg throughput\n")
    for N in n_values_used:
        matrices = generate_synthmatrix_names(N)
        if(N == 16):
            matrices = organicmatrices16 + matrices
        for matrix in matrices:
            demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
            demandMatrix = fct.return_normalized_matrix(demandMatrix)
            oneIteration(N, demandMatrix, matrix, outputfile)
