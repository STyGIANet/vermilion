#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:49:06 2024

@author: vamsi
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
#%%
directory = "/home/vamsi/src/phd/writings/rdcn-throughput/master/plots/"

heatmaps=["vision.png","image-processing.png","object-tracking.png","speech-recognition.png"]

# Function to convert RGB to hexadecimal
def rgb_to_hex(rgb):
    return '{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

visionMatrix=0
ipMatrix=0
otMatrix=0
srMatrix=0

# Load the heatmap image
for heatmap in heatmaps:
    heatmap_image_path = directory+heatmap
    heatmap_image = cv2.imread(heatmap_image_path)
    heatmap_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
    
    # Initialize the matrix to store the hexadecimal values
    if heatmap=="vision.png":
        rows, cols = 8, 8  # As specified for the heatmap
        visionMatrix = np.empty((rows, cols))
    if heatmap=="image-processing.png":
        rows, cols = 8, 8  # As specified for the heatmap
        ipMatrix = np.empty((rows, cols))
    if heatmap=="speech-recognition.png":
        rows, cols = 12,12
        srMatrix = np.empty((rows, cols))
    if heatmap=="object-tracking.png":
        rows,cols=9,9
        otMatrix = np.empty((rows, cols))
    cell_height = heatmap_rgb.shape[0] // rows
    cell_width = heatmap_rgb.shape[1] // cols
    hex_matrix = np.empty((rows, cols))
    
    # Iterate over the grid and convert each cell's color to hexadecimal
    for i in range(rows):
        for j in range(cols):
            # Extract the cell from the heatmap
            cell = heatmap_rgb[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width, :]
            # Compute the average color of the cell
            average_color = np.mean(np.mean(cell, axis=0), axis=0)
            # Convert the color to hexadecimal
            hex_color = rgb_to_hex(average_color)
            # Assign the hexadecimal value to the matrix cell
            hex_matrix[i, j] = 0xffffff - int(hex_color,16)
    
    hex_matrix = hex_matrix/np.max(hex_matrix)
    
    if heatmap=="vision.png":
        visionMatrix = hex_matrix
    if heatmap=="image-processing.png":
        ipMatrix = hex_matrix
    if heatmap=="speech-recognition.png":
        srMatrix = hex_matrix
    if heatmap=="object-tracking.png":
        otMatrix = hex_matrix
    
    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(hex_matrix, cmap='GnBu', linewidths=2)
    # plt.imshow(normalized_matrix/16252144, cmap='GnBu')
    # plt.colorbar()  # Show the color scale
    # fig.tight_layout()

#%%
rows, columns = 8,8
sums=list()
for i in range(rows):
    sums.append(sum(visionMatrix[i]))
for j in range(columns):
    sums.append(sum(visionMatrix[:,j]))
visionMatrixScaled = visionMatrix*8.0/np.max(sums)
visionMatrixFloor = visionMatrixScaled
for i in range(rows):
    for j in range(columns):
        visionMatrixFloor[i][j] = np.floor(visionMatrixScaled[i][j])
        
for i in range(rows):
    print("vision row "+str(i)+" sum: "+str(sum(visionMatrixFloor[i])))
for j in range(columns):
    print("vision column "+str(j)+" sum: "+str(sum(visionMatrixFloor[:,j])))
    
fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(visionMatrixFloor, cmap='GnBu', linewidths=2)
#%%
rows, columns = 8,8
sums=list()
for i in range(rows):
    sums.append(sum(ipMatrix[i]))
for j in range(columns):
    sums.append(sum(ipMatrix[:,j]))
ipMatrixScaled = ipMatrix*8.0/np.max(sums)
ipMatrixFloor = ipMatrixScaled
for i in range(rows):
    for j in range(columns):
        ipMatrixFloor[i][j] = np.floor(ipMatrixScaled[i][j])
        
for i in range(rows):
    print("IP row "+str(i)+" sum: "+str(sum(ipMatrixFloor[i])))
for j in range(columns):
    print("IP column "+str(j)+" sum: "+str(sum(ipMatrixFloor[:,j])))

fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(ipMatrixFloor, cmap='GnBu', linewidths=2)

#%%
rows, columns = 9,9
sums=list()
for i in range(rows):
    sums.append(sum(otMatrix[i]))
for j in range(columns):
    sums.append(sum(otMatrix[:,j]))
otMatrixScaled = otMatrix*8.0/np.max(sums)
otMatrixFloor = otMatrixScaled
for i in range(rows):
    for j in range(columns):
        otMatrixFloor[i][j] = np.floor(otMatrixScaled[i][j])
        
for i in range(rows):
    print("OT row "+str(i)+" sum: "+str(sum(otMatrixFloor[i])))
for j in range(columns):
    print("OT column "+str(j)+" sum: "+str(sum(otMatrixFloor[:,j])))

fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(otMatrixFloor, cmap='GnBu', linewidths=2)

#%%
rows, columns = 12,12
sums=list()
for i in range(rows):
    sums.append(sum(srMatrix[i]))
for j in range(columns):
    sums.append(sum(srMatrix[:,j]))
srMatrixScaled = srMatrix*8.0/np.max(sums)
srMatrixFloor = srMatrixScaled
for i in range(rows):
    for j in range(columns):
        srMatrixFloor[i][j] = np.floor(srMatrixScaled[i][j])
        
for i in range(rows):
    print("SR row "+str(i)+" sum: "+str(sum(srMatrixFloor[i])))
for j in range(columns):
    print("SR column "+str(j)+" sum: "+str(sum(srMatrixFloor[:,j])))
fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(srMatrixFloor, cmap='GnBu', linewidths=2)


#%%


############################################################################################

# TopoOpt DLRM Figure 1.


############################################################################################

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

directory = "/home/vamsi/src/phd/writings/rdcn-throughput/master/plots/"

matplotlib.rcParams.update({'font.size': 28})

heatmaps=["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3","topoopt"]
# heatmaps=["data-parallelism","hybrid-parallelism","heatmap2","heatmap3"]
# heatmaps=["hybrid-parallelism"]
# Function to convert RGB to hexadecimal
def rgb_to_hex(rgb):
    return '{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

visionMatrix=0
ipMatrix=0
otMatrix=0
srMatrix=0

maxEntry=0

# Load the heatmap image
for heatmap in heatmaps:
    heatmap_image_path = directory+heatmap+".png"
    heatmap_image = cv2.imread(heatmap_image_path)
    heatmap_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
    rows, cols = 16, 16  # As specified for the heatmap
    
    # # Initialize the matrix to store the hexadecimal values
    cell_height = heatmap_rgb.shape[0] // rows
    cell_width = heatmap_rgb.shape[1] // cols
    hex_matrix = np.empty((rows, cols))
    hex_matrixLog = np.empty((rows,cols))
    # Iterate over the grid and convert each cell's color to hexadecimal
    for i in range(rows):
        for j in range(cols):
            # Extract the cell from the heatmap
            cell = heatmap_rgb[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width, :]
            # Compute the average color of the cell
            average_color = np.mean(np.mean(cell, axis=0), axis=0)
            # Convert the color to hexadecimal
            hex_color = rgb_to_hex(average_color)
            # Assign the hexadecimal value to the matrix cell
            hex_matrix[i, j] = (0xffffff - int(hex_color,16))
            hex_matrixLog[i,j] = (hex_matrix[i,j]/0xffffff)**4
    maxEntry = np.max([maxEntry,np.sum(hex_matrixLog)])
    # print(maxEntry,np.max(hex_matrixLog),np.sum(hex_matrix), np.sum(hex_matrixLog))
    # hex_matrix = (np.exp(hex_matrix/np.max(hex_matrix))/np.exp(1))*44
    hex_matrix = (hex_matrix/np.max(hex_matrix))*44
    hex_matrixLog = ((hex_matrixLog))*np.sum(hex_matrixLog)**2

    ticks=[0.001, 0.04,0.4, 4, 44]
    
    fig = plt.figure(figsize=(8, 6))
    ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
    norm=colors.LogNorm(vmin=0.001,vmax=44,clip=True)
    ax = sns.heatmap(hex_matrixLog, cmap='GnBu',norm=colors.LogNorm(vmin=0.001,vmax=44,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=0.004,vmax=44)
    ax.collections[0].colorbar.set_ticklabels(ticklabels)
    ax.set_xticks([0,4,8,12])
    ax.set_xticklabels(["0","4","8","12"])
    ax.set_yticks([0,4,8,12])
    ax.set_yticklabels(["0","4","8","12"],rotation=0)
    # ax.collections[0].colorbar.set_ticklabels(ticklabels)
    # plt.imshow(normalized_matrix/16252144, cmap='GnBu')
    # plt.colorbar()  # Show the color scale
    fig.tight_layout()
    fig.savefig(directory+heatmap+'-DM.pdf')
    
    ###### Normalizing and scaling by 16 (number of nodes)
    maxRow = [0 for i in range(16)]
    maxColumn = [0 for i in range(16)]
    for i in range(16):
        for j in range(16):
            maxRow[i] = maxRow[i]+hex_matrixLog[i,j]
            maxColumn[j] = maxColumn[j] + hex_matrixLog[i,j]
    maxValue=np.max([maxRow,maxColumn])
    
    degree = 4
    normMatrix = hex_matrixLog
    
    ticks=[0, 0.01*degree/100, 0.1*degree/100, 1*degree/100, 100*degree/100]
    ticklabels=["0.001%","0.01%","0.1%","1%","100%"]
    
    figNorm = plt.figure(figsize=(8,6))
    ax = sns.heatmap(normMatrix, cmap='GnBu',linewidths=2, cbar_kws={'ticks': ticks},vmin=0.001,vmax=degree,norm=colors.LogNorm(vmin=0.001,vmax=degree,clip=True))
    ax.collections[0].colorbar.set_ticklabels(ticklabels)
    ax.set_xticks([0,4,8,12])
    ax.set_xticklabels(["0","4","8","12"])
    ax.set_yticks([0,4,8,12])
    ax.set_yticklabels(["0","4","8","12"],rotation=0)
    figNorm.tight_layout()
    figNorm.savefig(directory+heatmap+'-norm-DM.pdf')
    
    sumRow = [0 for i in range(16)]
    sumColumn = [0 for i in range(16)]
    for i in range(16):
        for j in range(16):
            sumRow[i] = sumRow[i] + normMatrix[i,j]
            sumColumn[j] = sumColumn[j] + normMatrix[i,j]
    
    print("####################################")
    print(heatmap)
    print("####################################")
    # print("original", np.min(sumColumn),np.min(sumRow),np.max(sumColumn),np.max(sumRow))
    
    
    floorMatrix=np.floor(normMatrix)

    sumFloorRow = [0 for i in range(16)]
    sumFloorColumn = [0 for i in range(16)]
    for i in range(16):
        for j in range(16):
            sumFloorRow[i] = sumFloorRow[i] + floorMatrix[i,j]
            sumFloorColumn[j] = sumFloorColumn[j] + floorMatrix[i,j]
    # print("floor", np.min(sumFloorColumn),np.min(sumFloorRow),np.max(sumFloorColumn),np.max(sumFloorRow))
    if (np.min(sumFloorColumn)< 1/2 and np.min(sumFloorRow) < 1/2 and np.max(sumFloorColumn) < 1/2 and np.max(sumFloorRow) < 1/2):
        print("ALL GOOD Interval 1")
    elif ((np.min(sumFloorColumn)>=1/2 and np.min(sumFloorColumn)< 3/4) and (np.min(sumFloorRow) >= 1/2 and np.min(sumFloorRow) < 3/4) and (np.max(sumFloorColumn) >= 1/2 and np.max(sumFloorColumn) < 3/4) and (np.max(sumFloorRow) >1/2 and np.max(sumFloorRow) < 3/4)):
        print("ALL GOOD Interval 2")
    elif (np.min(sumFloorColumn) >= 3/4 and np.min(sumFloorRow) >= 3/4 and np.max(sumFloorColumn) >= 3/4 and np.max(sumFloorRow) >= 3/4):
        print("ALL GOOD Interval 3")
    else:
        print("something wrong")
    
    ratiosRow=[0 for i in range(16)]
    ratiosColumn=[0 for i in range(16)]
    for i in range(16):
        x= sumFloorRow[i]/sumRow[i]
        ratiosRow[i] = x
        print("row", i , x)
        y =sumFloorColumn[i]/sumColumn[i]
        ratiosColumn[i]=y
        print("column", i , y)
    print("floor", np.min(ratiosRow),np.min(ratiosColumn),np.max(ratiosRow),np.max(ratiosColumn))
    
    residualMatrix=normMatrix-floorMatrix
    # for i in range(16):
    #     for j in range(16):
    #         residualMatrix[i,j] = 100*residualMatrix[i,j]/np.max([sumRow[i],sumColumn[j]])
    
    figFloor = plt.figure(figsize=(8,6))
    ax = sns.heatmap(floorMatrix, cmap='GnBu',linewidths=2, cbar_kws={'ticks': ticks},vmin=1,vmax=degree,norm=colors.LogNorm(vmin=0.001,vmax=degree,clip=True))
    ax.collections[0].colorbar.set_ticklabels(ticklabels)
    ax.set_xticks([0,4,8,12])
    ax.set_xticklabels(["0","4","8","12"])
    ax.set_yticks([0,4,8,12])
    ax.set_yticklabels(["0","4","8","12"],rotation=0)
    figFloor.tight_layout()
    figFloor.savefig(directory+heatmap+'-floor-DM.pdf')
    
    figResidual = plt.figure(figsize=(8,6))
    ax = sns.heatmap(residualMatrix, cmap='GnBu',linewidths=2, cbar_kws={'ticks': ticks},vmin=1,vmax=degree,norm=colors.LogNorm(vmin=0.001,vmax=degree,clip=True))
    ax.collections[0].colorbar.set_ticklabels(ticklabels)
    ax.set_xticks([0,4,8,12])
    ax.set_xticklabels(["0","4","8","12"])
    ax.set_yticks([0,4,8,12])
    ax.set_yticklabels(["0","4","8","12"],rotation=0)
    figResidual.tight_layout()
    figResidual.savefig(directory+heatmap+'-residual-DM.pdf')
    # floorRow = [0 for i in range(16)]
    # floorColumn = [0 for i in range(16)]
    # residualRow = [0 for i in range(16)]
    # residualColumn = [0 for i in range(16)]
    # for i in range(16):
    #     for j in range(16):
    #         floorRow[i] = floorRow[i]+np.floor(normMatrix[i,j])
    #         floorColumn[j] = floorColumn[j]+np.floor(normMatrix[i,j])
    #         residualRow[i] = residualRow[i]+normMatrix[i,j]-np.floor(normMatrix[i,j])
    #         residualColumn[j] = residualColumn[j]+normMatrix[i,j]-np.floor(normMatrix[i,j])
            

#%%

