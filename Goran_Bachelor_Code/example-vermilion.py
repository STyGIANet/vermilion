#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 07:21:47 2024

@author: vamsi
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import Throughput_as_Function as fct
import Rounding_Draft as rd
import Floor_Draft as fd
import paperCode as pc
import pandas as pd
#%%
N=16
degree=4
k = 3
B = 25
#%%

# plot dir
plotdir = "/home/vamsi/src/phd/writings/rdcn-throughput/nsdi2025/figures/example/"

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
# print("mygrep,N,matrix,maxValue,networkType,degree,throughput")

# iterations=[1-i*0.01 for i in range(50)]

# for degree in degrees:
    # for matrixfile in matrices:

        

N=16
matrixfile="heatmap16-4-dlrm.csv"
df = pd.read_csv(workdir+matrixfile)

data = np.zeros((N,N))

for i in range(len(df)):
    src = df["group"][i]
    dst = df["variable"][i]
    demand = df["value"][i]
    data[src][dst] = demand

# eps = 1e-5
# data[data < eps] = 0

plt.rcParams.update({'font.size': 24})

#%%
# sumMax = 0
# for i in range(len(data)):
#     if sumMax < np.sum(data[i]):
#         sumMax = np.sum(data[i])

# data = data/sumMax
# data = data * 44

############## Original matrix ####################
ticks=[1000, 10*10**3,100*10**3, 1000*10**3]
ticklabels=["1MB","10 MB","100 MB","1 GB"]
minVal = 1000
maxVal = 2*10**6

fig = plt.figure(figsize=(8, 6))
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)

ax = sns.heatmap(data, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
# ax = sns.heatmap(data, cmap='GnBu')
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-original.pdf")

#%%

plt.rcParams.update({'font.size': 32})

############## Normalized matrix ####################
sumMax = 0
for i in range(len(data)):
    if sumMax < np.sum(data[i]):
        sumMax = np.sum(data[i])

normalized_matrix = data/sumMax

ticks=[0.001, 0.01,0.1, 1]
ticklabels=["0.001","0.01","0.1","1"]

fig = plt.figure(figsize=(8, 6))
# ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
minVal = 1e-4
maxVal = 1
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(normalized_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True), cbar_kws={'ticks': ticks},linewidths=2,vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)


fig.tight_layout()
fig.savefig(plotdir+"dlrm-normalized.pdf")
#%%
############## Upscaled matrix ####################
upscaled_matrix = normalized_matrix*(k-1)*N

ticks=[0.001, 0.01,0.1, 1, k*N]
ticklabels=["0.001","0.01","0.1","1", "48"]

fig = plt.figure(figsize=(8, 6))
# ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
minVal = 1e-4
maxVal = k*N
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(upscaled_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True), cbar_kws={'ticks': ticks},linewidths=2,vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-upscaled.pdf")
#%%
############## Rounded matrix ####################
rounded_matrix = pc.generalized_rounding(normalized_matrix*(k-1)*N, N, 0)

ticks=[0.001, 0.01,0.1, 1, k*N]
ticklabels=["0.001","0.01","0.1","1", "48"]
fig = plt.figure(figsize=(8, 6))
# ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
minVal = 1e-4
maxVal = k*N
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(rounded_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True), cbar_kws={'ticks': ticks},linewidths=2,vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-rounded.pdf")

#%%
############## Complete Augmented matrix ####################
complete_augmented = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        complete_augmented[i][j] = rounded_matrix[i][j] + 1

ticks=[0.001, 0.01,0.1, 1, k*N]
ticklabels=["0.001","0.01","0.1","1", "48"]
fig = plt.figure(figsize=(8, 6))
# ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
minVal = 1e-4
maxVal = k*N
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(complete_augmented, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True), cbar_kws={'ticks': ticks},linewidths=2,vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-completeaug.pdf")
#%%
out_Left  = []
in_Left = []
for row in range(N):
    out_Left.append(int(k*N - np.sum(rounded_matrix[row,:])))
for col in range(N):
    in_Left.append(int(k*N - np.sum(rounded_matrix[:,col])))

G = nx.directed_configuration_model(in_Left, out_Left)

nx.draw(G)
plt.savefig(plotdir+"configuration-model-graph.pdf")
#%%
configuration_augmented = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        configuration_augmented[i][j] = G.number_of_edges(i,j) + complete_augmented[i][j]

ticks=[0.001, 0.01,0.1, 1, k*N]
ticklabels=["0.001","0.01","0.1","1", "48"]
fig = plt.figure(figsize=(8, 6))
# ticklabels=["0","0.04 GB","0.4 GB","4 GB","44 GB"]
minVal = 1e-4
maxVal = k*N
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(configuration_augmented, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True), cbar_kws={'ticks': ticks},linewidths=2,vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-configurationaug.pdf")
#%%
plt.rcParams.update({'font.size': 24})
sumMax = 0
for i in range(len(configuration_augmented)):
    if sumMax < np.sum(configuration_augmented[i,:]):
        sumMax = np.sum(configuration_augmented[i,:])

for i in range(len(configuration_augmented)):
    if sumMax < np.sum(configuration_augmented[:,i]):
        sumMax = np.sum(configuration_augmented[:,i])

renormalized_matrix = (configuration_augmented) * B / (k*N/degree)
ticks=[4, 6, 10, 15, 20, 25]
ticklabels=["4Gbps", "6Gbps", "10Gbps","15Gbps", "20Gbps", "25Gbps"]
fig = plt.figure(figsize=(8, 6))
minVal = 4
maxVal = B
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(renormalized_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)

fig.tight_layout()
fig.savefig(plotdir+"dlrm-emulated.pdf")


#%%

plt.rcParams.update({'font.size': 24})

oblivious = np.ones((N,N))
renormalized_oblivious_matrix = (oblivious) * B / (N/degree)
ticks=[4, 6, 10, 15, 20, 25]
ticklabels=["4Gbps", "6Gbps", "10Gbps","15Gbps", "20Gbps", "25Gbps"]
fig = plt.figure(figsize=(8, 6))
minVal = 4
maxVal = B
norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True)
ax = sns.heatmap(renormalized_oblivious_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
ax.collections[0].colorbar.set_ticklabels(ticklabels)
ax.set_xticks([0,4,8,12])
ax.set_xticklabels(["0","4","8","12"])
ax.set_yticks([0,4,8,12])
ax.set_yticklabels(["0","4","8","12"],rotation=0)


fig.tight_layout()
fig.savefig(plotdir+"dlrm-emulated-oblivious.pdf")