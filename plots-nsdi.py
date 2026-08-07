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

plotsdir = "/home/vamsi/src/phd/writings/rdcn-throughput/nsdi2025/plots/"
directory="/home/vamsi/src/phd/codebase/rdcn-throughput/"

#%%

# Manual color selection for bars
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
# Hatches for further distinction among bars
hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']

plt.rcParams.update({'font.size': 14})
# Load the updated dataset
df = pd.read_csv(directory+'Goran_Bachelor_Code/nsdi-throughput-results.txt',delimiter=" ")

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
df = pd.read_csv(directory+'Goran_Bachelor_Code/nsdi-throughput-results.txt',delimiter=" ")

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
df = pd.read_csv(directory+'Goran_Bachelor_Code/nsdi-throughput-results.txt',delimiter=" ")

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

df = pd.read_csv(directory+'Goran_Bachelor_Code/nsdi-throughput-results.txt',delimiter=" ")

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

df = pd.read_csv(directory+'Goran_Bachelor_Code/nsdi-throughput-results.txt',delimiter=" ")

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

#%%

algs=["rotornet","opera","complete","vermilion-3"]

algnames={}
algnames["rotornet"]="RotorNet"
algnames["opera"]="Opera"
algnames["vermilion-3"]="Vermilion"
algnames["complete"]="VLB (Sirius)"

markers={}
markers["rotornet"]="x"
markers["opera"]="s"
markers["vermilion-3"]="^"
markers["complete"]="o"

colors={}
colors["rotornet"]='brown'
colors["opera"]='green'
colors["vermilion-3"]='red'
colors["complete"]='blue'

results="/home/vamsi/lakewood/Mars-RDCN/results-vermilion/"

K=1000
M=K*K

loads=[0.05,0.1,0.2,0.4,0.6]
loadstr=["0.05","0.1","0.2","0.4","0.6"]

dms=["perm"]

plt.rcParams.update({'font.size': 14})

cdfindex="0"
#%%

########################################################################

# FCT PLOTS

#######################################################################

cdfindex="0"
flowStep = [ 0,5*K, 10*K, 20*K, 30*K, 50*K, 75*K, 100*K, 200*K, 400*K,600*K,800*K, 1*M, 5*M, 10*M,30*M  ]
flowSteps= [ 5*K, 10*K, 20*K, 30*K, 50*K, 75*K, 100*K, 200*K, 400*K,600*K,800*K, 1*M, 5*M, 10*M,30*M ]
fS=np.arange(len(flowSteps))
# flowSteps= [ "5K", "", "20K","", "50K", "", "100K","", "400K","","800K","", "5M", "","30M" ]

fctDict={}

plt.rcParams.update({'font.size': 14})

for dm in ["perm"]:
    fctDict[dm]={}
    for load in loads:
        fctDict[dm][load]={}
        for alg in algs:
            fctDict[dm][load][alg]=list()
            df=pd.read_csv(results+str(alg)+'-'+str(cdfindex)+'-'+str(dm)+'-'+str(load)+'.fct',delimiter=' ',usecols=[3,4],names=["size","fct"])
            lfct99=list()
            lfct95=list()
            lfct50=list()
            for i in range(1,len(flowStep)):
                df1=df.loc[ (df["size"]<flowStep[i]) & (df["size"] >= flowStep[i-1]) ]
                fcts=df1["fct"].to_list()
                sd=fcts
                sd.sort()
                try:
                    fct99 = sd[int(len(sd)*0.999)]
                except:
                    fct99 = 10**6
                try:
                    fct95 = sd[int(len(sd)*0.95)]
                except:
                    fct95 = 10**6
                try:
                    fct50 = sd[int(len(sd)*0.5)]
                except:
                    fct50 = 10**6
                # print(fct99,fct95,fct50,flowStep[i],load,alg)
                lfct99.append(fct99)
                lfct95.append(fct95)
                lfct50.append(fct50)
            fctDict[dm][load][alg]=lfct99
        
#%%

cdfindex="0"
plt.rcParams.update({'font.size': 16})
flowStep = [ 0,5*K, 10*K, 20*K, 30*K, 50*K, 75*K, 100*K, 200*K, 400*K,600*K,800*K, 1*M, 5*M, 10*M,30*M  ]
flowSteps= [ 5*K, 10*K, 20*K, 30*K, 50*K, 75*K, 100*K, 200*K, 400*K,600*K,800*K, 1*M, 5*M, 10*M,30*M ]
fS=np.arange(len(flowSteps))
# flowSteps= [ "5K", "", "20K","", "50K", "", "100K","", "400K","","800K","", "5M", "","30M" ]


for dm in ["perm"]:
    for load in loads:
        fig,ax = plt.subplots(1,1,figsize=(4,6))
        for alg in algs:
            ax.plot(flowSteps,fctDict[dm][load][alg],label=algnames[alg],marker=markers[alg],c=colors[alg],lw=2,markersize=8)
            # ax1.plot(flowSteps,lfct50,label=algnames[alg],marker=markers[alg],c=colors[alg],lw=2)
 
        # ax.legend()
        ax.set_xscale('log')
        ax.set_ylim(0.001,2*10**3)
        ax.set_yscale('log')
        ax.set_yticks([0.01,1,100])
        ax.set_yticklabels(["0.01","1","100"])
        ax.set_ylabel("99-pct FCT (ms)",fontsize=20)
        ax.set_xlabel("Flow size (Bytes)",fontsize=20)
        
        # ax.set_title('load='+str(load))
        
        ax.set_xticks([10**4, 10**5, 10**6, 10**7])
        ax.set_xticklabels(["10KB","100KB","1MB","10MB"],rotation=35)
        
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        fig.tight_layout()
        # fig.savefig(plotsdir+'fcts-'+str(load)+'-'+str(dm)+'.pdf')
        fig.savefig(plotsdir+'fcts-'+str(load)+'-'+str(dm)+'-'+str(cdfindex)+'.pdf')
    
#%%
algs=["rotornet","opera","complete","vermilion-3"]


print ("########### SHORT FLOWS #########")
for dm in dms:
    for load in loads:
        for alg in algs:
            diff = (fctDict[dm][load][alg][0]-fctDict[dm][load]["vermilion-3"][0])/(fctDict[dm][load][alg][0])
            print(dm,load,alg,fctDict[dm][load][alg][0],fctDict[dm][load]["vermilion-3"][0],diff*100)
            
print ("########### Long FLOWS #########")
for dm in dms:
    for load in loads:
        for alg in algs:
            diff = (fctDict[dm][load][alg][-1]-fctDict[dm][load]["vermilion-3"][-1])/(fctDict[dm][load][alg][-1])
            print(dm,load,alg,fctDict[dm][load][alg][-1],fctDict[dm][load]["vermilion-3"][-1],diff*100)


#%%
algs=["rotornet","opera","complete","vermilion-3"]
cdfindex="0"

thDict={}
plt.rcParams.update({'font.size': 16})
for dm in dms:
    fig,ax = plt.subplots(1,1,figsize=(4,5))
    thDict[dm]={}
    for alg in algs:
        thDict[dm][alg]=list()
        thlist=list()
        for load in loads:
            df=pd.read_csv(results+str(alg)+'-'+str(cdfindex)+'-'+str(dm)+'-'+str(load)+'.throughput',delimiter=' ',usecols=[0,1],names=["throughput","time"])
            th=df["throughput"].to_list()
            avth=np.mean(th)
            thlist.append(avth*100)
        thDict[dm][alg]=thlist
        ax.plot([i*100 for i in loads],thlist,label=algnames[alg],marker=markers[alg],c=colors[alg],lw=2.2,markersize=11)
    
    ax.legend()
    ax.set_ylim(0,70)
    ax.set_ylabel("Link utilization (%)")
    ax.set_xlabel("Load (%)")
    # ax.set_xscale('log')
    # ax.set_ylim(0.05,2*10**3)
    # ax.set_yscale('log')
    
    # ax.set_title('load='+str(load))
    
    ax.xaxis.grid(True,ls='--')
    ax.yaxis.grid(True,ls='--')
    fig.tight_layout()
    # fig.savefig(plotsdir+'th-'+str(dm)+'.pdf')
    fig.savefig(plotsdir+'th-'+str(dm)+'-'+str(cdfindex)+'.pdf')
    
#%%
algs=["rotornet","opera","complete","vermilion-3"]

print ("########### Throughput #########")
for dm in dms:
    for alg in algs:
        for i in range(len(loads)):
            diff = (thDict[dm][alg][i]-thDict[dm]["vermilion-3"][i])/(thDict[dm][alg][i])
            print(dm,loads[i],alg,thDict[dm][alg][i],thDict[dm]["vermilion-3"][i],diff*100, thDict[dm]["vermilion-3"][i]/thDict[dm][alg][i])

#%%


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