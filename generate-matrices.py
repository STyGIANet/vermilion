import matplotlib.pyplot as plt
# import cv2
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from Goran_Bachelor_Code import random_Matrices as RM

directory = "/home/vamsi/src/phd/writings/rdcn-throughput/master/plots/"
workdir="./matrices/"
heatmaps=["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3","topoopt"]
#%%
# Function to convert RGB to hexadecimal
def rgb_to_hex(rgb):
    return '{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# for heatmap in heatmaps:
#     heatmap_image_path = directory+heatmap+".png"
#     heatmap_image = cv2.imread(heatmap_image_path)
#     heatmap_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
#     rows, cols = 16, 16  # As specified for the heatmap

#     # # Initialize the matrix to store the hexadecimal values
#     cell_height = heatmap_rgb.shape[0] // rows
#     cell_width = heatmap_rgb.shape[1] // cols
#     hex_matrix = np.empty((rows, cols))
#     hex_matrixLog = np.empty((rows,cols))
#     maxEntry=0
#     # Iterate over the grid and convert each cell's color to hexadecimal
#     for i in range(rows):
#         for j in range(cols):
#             # Extract the cell from the heatmap
#             cell = heatmap_rgb[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width, :]
#             # Compute the average color of the cell
#             average_color = np.mean(np.mean(cell, axis=0), axis=0)
#             # Convert the color to hexadecimal
#             hex_color = rgb_to_hex(average_color)
#             # Assign the hexadecimal value to the matrix cell
#             hex_matrix[i, j] = (0xffffff - int(hex_color,16))
#             hex_matrixLog[i,j] = (hex_matrix[i,j]/0xffffff)**4
#     maxEntry = np.max([maxEntry,np.sum(hex_matrixLog)])
#     # print(maxEntry,np.max(hex_matrixLog),np.sum(hex_matrix), np.sum(hex_matrixLog))
#     # hex_matrix = (np.exp(hex_matrix/np.max(hex_matrix))/np.exp(1))*44
#     hex_matrix = (hex_matrix/np.max(hex_matrix))*44
#     hex_matrixLog = ((hex_matrixLog))*np.sum(hex_matrixLog)**2

#     ###### Normalizing and scaling by 16 (number of nodes)
#     maxRow = [0 for i in range(16)]
#     maxColumn = [0 for i in range(16)]
#     for i in range(16):
#         for j in range(16):
#             maxRow[i] = maxRow[i]+hex_matrixLog[i,j]
#             maxColumn[j] = maxColumn[j] + hex_matrixLog[i,j]
#     maxValue=np.max([maxRow,maxColumn])

#     normMatrix = hex_matrixLog/maxValue

#     outputfile = open(workdir+heatmap+".mat", "w")
#     for i in range(16):
#         for j in range(16):
#             outputfile.write(str(normMatrix[i,j])+" ")
#         outputfile.write("\n")
#     outputfile.close()

#####################################################################################
 #%%
# for N in [8,16,32,48,64,128,512,256,1024]:
for N in [512]:
    ######### Chessboard ######## 
    matrixname="chessboard-" + str(N)
    outputfile = open(workdir+matrixname+".mat", "w")
    demand = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                demand[i, j] = 0.5 if (i + j) % 2 == 0 else 1.5
                outputfile.write(str(demand[i,j]/(N-1))+" ")
            else:
                outputfile.write("0.0 ")
        outputfile.write("\n")
    outputfile.close()

#     ######### Uniform ######## 
    outputfile = open(workdir+"uniform-"+str(N)+".mat", "w")
    uniformdemand = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i!= j:
                uniformdemand[i, j] = 1
                outputfile.write(str(uniformdemand[i,j]/(N-1))+" ")
            else:
                outputfile.write("0.0 ")
        outputfile.write("\n")
    outputfile.close()

    ######### Permutation ######## 
    outputfile = open(workdir+"permutation-"+str(N)+".mat", "w")
    permdemand = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                permdemand[i, j] = N-1 if j==(i+N/2)%N else 0
                outputfile.write(str(permdemand[i,j]/(N-1))+" ")
            else:
                outputfile.write("0.0 ")
        outputfile.write("\n")
    outputfile.close()

    ######### Uniform + Permutation ######## 
    for skew in [i/10 for i in range(1,10)]:
        matrixname="skew-"+str(N)+"-"+str(skew)
        outputfile = open(workdir+matrixname+".mat", "w")
        skewdemand = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i!= j:
                    skewdemand[i, j] = (1-skew)*uniformdemand[i,j] + skew*permdemand[i,j]
                    outputfile.write(str(skewdemand[i,j]/(N-1))+" ")
                else:
                    outputfile.write("0.0 ")
            outputfile.write("\n")
        outputfile.close()
#####################################################################################
#Random Matrices

# N = 64
# matrixname="random-skewed-" + str(N)
# outputfile = open(workdir+matrixname+".mat", "w")
# demand = RM.generate_doubly_stochastic_matrix(N, min_val=1e-5)
# print(np.array2string(demand,formatter={'float_kind':lambda x: "%.4f" % x}))
# for i in range(N):
#     for j in range(N):
#         if i!=j:
#             outputfile.write(str(demand[i,j])+" ")
#         else:
#             outputfile.write("0.0 ")
#     outputfile.write("\n")
# outputfile.close()