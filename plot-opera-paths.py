#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:07:56 2024

@author: vamsi
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plotsdir = "plots/"
directory="./"

#%%

# Manual color selection for bars
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
# Hatches for further distinction among bars
hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']

plt.rcParams.update({'font.size': 14})


####### Paths

df = pd.read_csv(directory+"opera-paths-64-16.csv",header=None,usecols=[0],names=["pathlen"])

pathlenopera=list(df["pathlen"])

pathlenopera.sort()

X = pathlenopera
Y = np.arange(len(pathlenopera))/(len(pathlenopera)-1)

fig,ax = plt.subplots(1,1,figsize=(4,5))
ax.plot(X,Y,c='green', lw=2)
ax.plot(X[-1],Y[-1],marker='s', markersize=10,c='green',label="Opera")
ax.plot([1 for i in range(100)], 0.01*np.arange(100) ,c='red',lw=2)
ax.plot(1,1,marker='d', markersize=10,c='red',label="Vermilion")

ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')
ax.set_ylabel("CDF")
ax.set_xlabel("Number of hops")
ax.legend()
fig.tight_layout()
fig.savefig(plotsdir+'opera-paths.pdf')