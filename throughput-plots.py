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
# Load the updated dataset
df = pd.read_csv(directory+'throughput-analysis/nsdi2027-throughput-results.csv',delimiter=" ")

# Filter data for N=16
df_n16 = df[df['n'] == 16]

# Separate out the rows as per the conditions specified
rotornet_k1 = df_n16[(df_n16['alg'] == 'rotornet') & (df_n16['k'] == 1) & (df_n16['matrix'] != "topoopt")]
vermilion_k3 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 3) & (df_n16['matrix'] != "topoopt")]
vermilion_k4 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 6) & (df_n16['matrix'] != "topoopt")]

# Prepare bar plot data
labels = list(rotornet_k1['matrix'])  # Assuming matrix is the identifier for comparison

labels = [  "Chessboard",
            "All-to-All",
            "Ring",
            "skew-0.1",
            "skew-0.2",
            "skew-0.3",
            "skew-0.4",
            "skew-0.5",
            "skew-0.6",
            "skew-0.7",
            "skew-0.8",
            "skew-0.9",
            "DLRM-data","DLRM-hybrid","DLRM-permute1", "DLRM-permute2"
]

# Create a list of throughput values
rotornet_throughput_sh = list(rotornet_k1['throughputSH'])
rotornet_throughput_mh = list(rotornet_k1['throughputMH'])
vermilion_k3_throughput_sh = list(vermilion_k3['throughputSH'])
vermilion_k4_throughput_sh = list(vermilion_k4['throughputSH'])

# Set the width of the bars
bar_width = 0.2

# Set positions of bar on X axis
r1 = range(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting
fig , ax = plt.subplots(1,1,figsize=(12, 5))

# Bar plot for rotornet (throughputSH and throughputMH)
ax.bar(r1, rotornet_throughput_sh, color=colors[0],hatch=hatches[0], width=bar_width, edgecolor='black', label='Oblivious (single-hop)',alpha=0.6)
ax.bar(r2, rotornet_throughput_mh, color=colors[1],hatch=hatches[1], width=bar_width, edgecolor='black', label='Oblivious (multi-hop)',alpha=0.6)

# Bar plot for vermilion k=3 (throughputSH) and k=4 (throughputSH)
ax.bar(r3, vermilion_k3_throughput_sh, color=colors[2],hatch=hatches[2], width=bar_width, edgecolor='black', label='Vermilion k=3',alpha=0.6)
ax.bar([x + bar_width for x in r3], vermilion_k4_throughput_sh, color=colors[3], hatch=hatches[3], width=bar_width, edgecolor='black', label='Vermilion k=6',alpha=0.6)

# Adding labels and title
# ax.set_xlabel('Demand matrix',fontsize=16)
ax.set_ylabel('Throughput',fontsize=16)

ax.set_xticks([r + bar_width for r in range(len(labels))], labels, rotation=25)
ax.set_xticklabels(labels)
ax.set_ylim(0,1.01)
ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')


# Move the legend outside
# fig.legend(bbox_to_anchor=(0.25, 1.05), loc='upper left',ncol=2)

# Adjust layout to make room for the legend
fig.tight_layout()
fig.savefig(plotsdir+"throughput-16.pdf")

#%%

for i in range(len(labels)):
    print(labels[i],rotornet_throughput_sh[i], rotornet_throughput_mh[i], vermilion_k3_throughput_sh[i], vermilion_k4_throughput_sh[i])


#%%

# Manual color selection for bars
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
# Hatches for further distinction among bars
hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']

plt.rcParams.update({'font.size': 14})
# Load the updated dataset
df = pd.read_csv(directory+'throughput-analysis/nsdi2027-throughput-results.csv',delimiter=" ")

# Filter data for N=16
df_n16 = df[df['n'] == 32]

# Separate out the rows as per the conditions specified
rotornet_k1 = df_n16[(df_n16['alg'] == 'rotornet') & (df_n16['k'] == 1)]
vermilion_k3 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 3)]
vermilion_k4 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 6)]

# Prepare bar plot data
labels = list(rotornet_k1['matrix'])  # Assuming matrix is the identifier for comparison

labels = [
            "Chessboard",
            "All-to-All",
            "Ring",
            "skew-0.1",
            "skew-0.2",
            "skew-0.3",
            "skew-0.4",
            "skew-0.5",
            "skew-0.6",
            "skew-0.7",
            "skew-0.8",
            "skew-0.9",]

# Create a list of throughput values
rotornet_throughput_sh = list(rotornet_k1['throughputSH'])
rotornet_throughput_mh = list(rotornet_k1['throughputMH'])
vermilion_k3_throughput_sh = list(vermilion_k3['throughputSH'])
vermilion_k4_throughput_sh = list(vermilion_k4['throughputSH'])

# Set the width of the bars
bar_width = 0.2

# Set positions of bar on X axis
r1 = range(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting
fig , ax = plt.subplots(1,1,figsize=(12, 5))

# Bar plot for rotornet (throughputSH and throughputMH)
ax.bar(r1, rotornet_throughput_sh, color=colors[0],hatch=hatches[0], width=bar_width, edgecolor='black', label='Oblivious (single-hop)',alpha=0.6)
ax.bar(r2, rotornet_throughput_mh, color=colors[1],hatch=hatches[1], width=bar_width, edgecolor='black', label='Oblivious (multi-hop)',alpha=0.6)

# Bar plot for vermilion k=3 (throughputSH) and k=4 (throughputSH)
ax.bar(r3, vermilion_k3_throughput_sh, color=colors[2],hatch=hatches[2], width=bar_width, edgecolor='black', label='Vermilion k=3',alpha=0.6)
ax.bar([x + bar_width for x in r3], vermilion_k4_throughput_sh, color=colors[3], hatch=hatches[3], width=bar_width, edgecolor='black', label='Vermilion k=6',alpha=0.6)

# Adding labels and title
# ax.set_xlabel('Demand matrix',fontsize=16)
ax.set_ylabel('Throughput',fontsize=16)

ax.set_xticks([r + bar_width for r in range(len(labels))], labels, rotation=45)
ax.set_xticklabels(labels)
ax.set_ylim(0,1.01)
ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')


# Move the legend outside
# fig.legend(bbox_to_anchor=(0.25, 1.05), loc='upper left',ncol=2)

# Adjust layout to make room for the legend
fig.tight_layout()
fig.savefig(plotsdir+"throughput-32.pdf")
#%%

# Manual color selection for bars
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
# Hatches for further distinction among bars
hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']

plt.rcParams.update({'font.size': 14})
# Load the updated dataset
df = pd.read_csv(directory+'throughput-analysis/nsdi2027-throughput-results.csv',delimiter=" ")

# Filter data for N=16
df_n16 = df[df['n'] == 48]

# Separate out the rows as per the conditions specified
rotornet_k1 = df_n16[(df_n16['alg'] == 'rotornet') & (df_n16['k'] == 1)]
vermilion_k3 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 3)]
vermilion_k4 = df_n16[(df_n16['alg'] == 'vermilion') & (df_n16['k'] == 6)]

# Prepare bar plot data
labels = list(rotornet_k1['matrix'])  # Assuming matrix is the identifier for comparison

labels = [
            "Chessboard",
            "All-to-All",
            "Ring",
            "skew-0.1",
            "skew-0.2",
            "skew-0.3",
            "skew-0.4",
            "skew-0.5",
            "skew-0.6",
            "skew-0.7",
            "skew-0.8",
            "skew-0.9",]

# Create a list of throughput values
rotornet_throughput_sh = list(rotornet_k1['throughputSH'])
rotornet_throughput_mh = list(rotornet_k1['throughputMH'])
vermilion_k3_throughput_sh = list(vermilion_k3['throughputSH'])
vermilion_k4_throughput_sh = list(vermilion_k4['throughputSH'])

# Set the width of the bars
bar_width = 0.2

# Set positions of bar on X axis
r1 = range(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting
fig , ax = plt.subplots(1,1,figsize=(12, 5))

# Bar plot for rotornet (throughputSH and throughputMH)
ax.bar(r1, rotornet_throughput_sh, color=colors[0],hatch=hatches[0], width=bar_width, edgecolor='black', label='Oblivious (single-hop)',alpha=0.6)
ax.bar(r2, rotornet_throughput_mh, color=colors[1],hatch=hatches[1], width=bar_width, edgecolor='black', label='Oblivious (multi-hop)',alpha=0.6)

# Bar plot for vermilion k=3 (throughputSH) and k=4 (throughputSH)
ax.bar(r3, vermilion_k3_throughput_sh, color=colors[2],hatch=hatches[2], width=bar_width, edgecolor='black', label='Vermilion k=3',alpha=0.6)
ax.bar([x + bar_width for x in r3], vermilion_k4_throughput_sh, color=colors[3], hatch=hatches[3], width=bar_width, edgecolor='black', label='Vermilion k=6',alpha=0.6)

# Adding labels and title
# ax.set_xlabel('Demand matrix',fontsize=16)
ax.set_ylabel('Throughput',fontsize=16)

ax.set_xticks([r + bar_width for r in range(len(labels))], labels, rotation=25)
ax.set_xticklabels(labels)
ax.set_ylim(0,1.01)
ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')


# Move the legend outside
# fig.legend(bbox_to_anchor=(0.25, 1.05), loc='upper left',ncol=2)

# Adjust layout to make room for the legend
fig.tight_layout()
fig.savefig(plotsdir+"throughput-48.pdf")




#%%

plt.rcParams.update({'font.size': 18})

df = pd.read_csv(directory+'throughput-analysis/nsdi2027-throughput-results.csv',delimiter=" ")

N=32
df = df[(df["n"]==N)&(df["alg"]=="vermilion")]

dfperm = df[df["matrix"]=="permutation-"+str(N)]
dfhybrid = df[df["matrix"]=="skew-"+str(N)+"-0.5"]

# Prepare the plot
fig, ax = plt.subplots(1,1,figsize=(4,6))

# Plot throughput vs. k
ax.plot(dfperm["k"], dfperm["throughputSH"], marker='o', color='green', label='Ring', lw=2, markersize=10)
ax.plot(dfhybrid["k"], dfhybrid["throughputSH"], marker='x', color='red', label='Skew-0.5', lw=2, markersize=10)

# Adding labels and title
ax.set_xlabel('Parameter k')
ax.set_ylabel('Throughput')

ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')

ax.set_ylim(0.4,1.01)

# Add a legend
ax.legend()

# Show the plot
fig.tight_layout()
fig.savefig(plotsdir+"throughput-k.pdf")

#%%

plt.rcParams.update({'font.size': 18})

df = pd.read_csv(directory+'throughput-analysis/nsdi2027-throughput-results.csv',delimiter=" ")

df = df[df["alg"]=="vermilion"]

dfperm3 = df[(df["matrix"].str.contains("permutation-"))&(df["k"]==3)]
dfperm6 = df[(df["matrix"].str.contains("permutation-"))&(df["k"]==6)]

dfhybrid3 = df[(df["matrix"].str.contains("skew-.*-0.5"))&(df["k"]==3)]
dfhybrid6 = df[(df["matrix"].str.contains("skew-.*-0.5"))&(df["k"]==6)]

# Prepare the plot
fig, ax = plt.subplots(1,1,figsize=(4,6))

# Plot throughput vs. k
ax.plot(np.arange(len(dfperm3["n"])), dfperm3["throughputSH"], marker='o', color='green', label='Ring (k=3)', lw=3, markersize=10)
ax.plot(np.arange(len(dfperm6["n"])), dfperm6["throughputSH"], marker='o', color='green', label='Ring (k=6)', lw=3, markersize=10, ls='--')
ax.plot(np.arange(len(dfhybrid3["n"])), dfhybrid3["throughputSH"], marker='x', color='red', label='Skew (k=3)', lw=3, markersize=10)
ax.plot(np.arange(len(dfhybrid6["n"])), dfhybrid6["throughputSH"], marker='x', color='red', label='Skew (k=6)', lw=3, markersize=10,ls='--')

ax.set_xticks(np.arange(len(dfperm3["n"])))
ax.set_xticklabels(dfperm3["n"],rotation=45)

# Adding labels and title
ax.set_xlabel('# Nodes')
ax.set_ylabel('Throughput')

ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')

ax.set_ylim(0.4,1.01)

ax.axhline(0.83,c='k',ls='--')

ax.axhline(0.66,c='k',ls='--')

# Add a legend
ax.legend()
# fig.legend(bbox_to_anchor=(1, 1.05),framealpha=0.8, ncol=2)

# Show the plot
fig.tight_layout()
fig.savefig(plotsdir+"throughput-nodes.pdf")