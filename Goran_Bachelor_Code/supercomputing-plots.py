#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:03:38 2024

@author: vamsi
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

plotsdir = "/home/vamsi/src/phd/writings/vermilion-part-2/supercomputing2025/plots/"
directory="/home/vamsi/rdcn-throughput/"

#%%

# Manual color selection for bars
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
# Hatches for further distinction among bars
hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']

plt.rcParams.update({'font.size': 18})
# Load the updated dataset
df = pd.read_csv(directory+'Goran_Bachelor_Code/sigmetrics-throughput-results-16.csv',delimiter=" ",names=["matrix","alg","k","n","d","noise","type","throughput"])
#%%
N = 16
# dfN = df[(df['n'] == N) & (df['alg']=="vermRobustness") & (df['matrix']!="random-skewed-"+str(N))]
dfN = df[(df['n'] == N) & (df['alg']=="vermThroughput")]

dfX = dfN[(dfN['k']==1)&(dfN['noise']==1)&(dfN['type']=="add")]
# print(dfX['matrix'])

matrices = list(dfX['matrix'])  # Assuming matrix is the identifier for comparison

labels = [  "Chessboard",
            "All-to-All",
            "Ring",
            "Random-skewed",
            "skew-0.2",
            "skew-0.4",
            "skew-0.6",
            "skew-0.8",
]

#%%
noises=np.arange(10)
kvalues = np.arange(10)
heatmap={}
for noisetype in ["add","mult"]:
    heatmap[noisetype]={}
    for matrix in matrices:
        heatmap[noisetype][matrix] = np.zeros((10,10))
        for noise in range(10):
            for k in range(10):
                heatmap[noisetype][matrix][noise][k] = list(dfN[ (dfN['matrix']==matrix) & (dfN['k']==k+1) & (dfN['noise']==noise) & (dfN['type']==noisetype) ]["throughput"])[0]
#%%
for noisetype in ["add","mult"]:
    for matrix in matrices:
        fig, ax = plt.subplots(1,1)
        cax = ax.imshow(heatmap[noisetype][matrix], cmap='CMRmap', interpolation='nearest', vmin=0.3, vmax=1)
    
        fig.colorbar(cax, ax=ax)
        
        ax.set_xticks(np.arange(1,11,2))
        ax.set_xticklabels(np.arange(2,11,2))
        
        ax.set_ylabel("Noise factor")
        ax.set_xlabel(r'$\gamma$'+" parameter")
        
        # ax.set_title(matrix+" "+noisetype)
        fig.tight_layout()
        plt.show()
        fig.savefig(plotsdir+"det-"+matrix+'-'+noisetype+'.pdf',bbox_inches='tight')
    
#%%

N = 16
# dfN = df[(df['n'] == N) & (df['alg']=="randRobustness") & (df['matrix']!="random-skewed-"+str(N))]
dfN = df[(df['n'] == N) & (df['alg']=="randThroughput")]

dfX = dfN[(dfN['k']==1)&(dfN['noise']==1)&(dfN['type']=="add")]

matrices = list(dfX['matrix'])  # Assuming matrix is the identifier for comparison

labels = [  "Chessboard",
            "All-to-All",
            "Ring",
            "Random-skewed",
            "skew-0.2",
            "skew-0.4",
            "skew-0.6",
            "skew-0.8",
]

#%%
noises=np.arange(10)
kvalues = np.arange(10)
heatmap={}
for noisetype in ["add","mult"]:
    heatmap[noisetype]={}
    for matrix in matrices:
        heatmap[noisetype][matrix] = np.zeros((10,10))
        for noise in range(10):
            for k in range(10):
                heatmap[noisetype][matrix][noise][k] = list(dfN[ (dfN['matrix']==matrix) & (dfN['k']==k+1) & (dfN['noise']==noise) & (dfN['type']==noisetype) ]["throughput"])[0]
#%%
for noisetype in ["add","mult"]:
    for matrix in matrices:
        fig, ax = plt.subplots(1,1)
        cax = ax.imshow(heatmap[noisetype][matrix], cmap='CMRmap', interpolation='nearest', vmin=0.3, vmax=1)
    
        fig.colorbar(cax, ax=ax)
        
        ax.set_xticks(np.arange(1,11,2))
        ax.set_xticklabels(np.arange(2,11,2))
        
        ax.set_ylabel("Noise factor")
        ax.set_xlabel(r'$\gamma$'+" parameter")
        
        # ax.set_title(matrix+" "+noisetype)
        fig.tight_layout()
        plt.show()
        fig.savefig(plotsdir+"rand-"+matrix+'-'+noisetype+'.pdf',bbox_inches='tight')