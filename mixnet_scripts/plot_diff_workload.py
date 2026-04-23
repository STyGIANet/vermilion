
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:41:00 2026

@author: deeksha-p-rao
"""

import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import seaborn as sns
#%%

def load_data(path):
    df = pd.read_csv(path, sep=r"\s+")
    
    df["algorithm"] = df["algorithm"].astype("category")
    df["workload"] = df["workload"].astype("category")
    df["bw"] = pd.to_numeric(df["bw"])
    df["rdelay"] = pd.to_numeric(df["rdelay"])
    df["time"] = pd.to_numeric(df["time"])
    
    return df

def build_nested_dict(df):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for _, row in df.iterrows():
        data[row["algorithm"]][row["bw"]][row["rdelay"]][row["workload"]] = row["time"]
    
    return data
#%%
df = load_data("results.txt")
data = build_nested_dict(df)


#%%

### Example

subset = df[(df["workload"] == "Phi-3.5-MoE-instruct-gsm8k") & (df["rdelay"] == 10)]


for alg in subset["algorithm"].unique():
    alg_data = subset[subset["algorithm"] == alg].sort_values("bw")
    plt.plot(alg_data["bw"], alg_data["time"], marker="o", label=alg)

plt.xlabel("Bandwidth (bw)")
plt.ylabel("Time")
plt.title("Time vs Bandwidth")
plt.legend()
plt.grid(True)
plt.show()


subset = df[(df["workload"] == "mixtral") & (df["rdelay"] == 10)]


for alg in subset["algorithm"].unique():
    alg_data = subset[subset["algorithm"] == alg].sort_values("bw")
    plt.plot(alg_data["bw"], alg_data["time"], marker="o", label=alg)

plt.xlabel("Bandwidth (bw)")
plt.ylabel("Time")
plt.title("Time vs Bandwidth for mixtral")
plt.legend()
plt.grid(True)
plt.show()


subset = df[(df["workload"] == "Phi-3.5-MoE-instruct-mmlu") & (df["rdelay"] == 10)]


for alg in subset["algorithm"].unique():
    alg_data = subset[subset["algorithm"] == alg].sort_values("bw")
    plt.plot(alg_data["bw"], alg_data["time"], marker="o", label=alg)

plt.xlabel("Bandwidth (bw)")
plt.ylabel("Time")
plt.title("Time vs Bandwidth for Phi-3.5-MoE-instruct-mmlu")
plt.legend()
plt.grid(True)
plt.show()


subset = df[(df["workload"] == "qwen") & (df["rdelay"] == 10)]


for alg in subset["algorithm"].unique():
    alg_data = subset[subset["algorithm"] == alg].sort_values("bw")
    plt.plot(alg_data["bw"], alg_data["time"], marker="o", label=alg)

plt.xlabel("Bandwidth (bw)")
plt.ylabel("Time")
plt.title("Time vs Bandwidth for qwen")
plt.legend()
plt.grid(True)
plt.show()


subset = df[(df["workload"] == "qwen_128") & (df["rdelay"] == 10)]


for alg in subset["algorithm"].unique():
    alg_data = subset[subset["algorithm"] == alg].sort_values("bw")
    plt.plot(alg_data["bw"], alg_data["time"], marker="o", label=alg)

plt.xlabel("Bandwidth (bw)")
plt.ylabel("Time")
plt.title("Time vs Bandwidth for qwen_128")
plt.legend()
plt.grid(True)
plt.show()


#%%

markers={}
markers["vermilion"]='^'
markers["birkhoff"]='x'
markers["rotor"]='o'
markers["mixnet"]='d'

colors={}
colors["vermilion"]='r'
colors["birkhoff"]='g'
colors["rotor"]='blue'
colors['mixnet']='k'

labels={}
labels["vermilion"]="Vermilion"
labels["birkhoff"]="BvN"
labels["rotor"]="VLB"
labels["mixnet"]="MixNet"

bandwidths = df["bw"].unique()
rdelays = df["rdelay"].unique()
workloads = df["workload"].unique()
algs = df["algorithm"].unique()

rdelaylabels=["10ns","100ns", "500ns",r'1$\mu$s',r'10$\mu$s',r'100$\mu$s',"1ms","10ms"]
bandwidthlabels=["800Gbps", "1.6Tbps", "2.4Tbps", "3.2Tbps", "4.8Tbps", "6.4Tbps", "8Tbps", "9.6Tbps"]

matplotlib.rcParams.update({'font.size': 22})


# heatmaps alg/vermilion (speedup) x --> rdelay ; y --> bandwidth

# one heatmap for each workload - 4
# one heatmap for each baseline - 3

# for workload in workloads:
#     arr = np.zeros()
#     for rdelay in rdelays:
#         fig, ax = plt.subplots(1,1,figsize=(5,3))
#         for alg in algs:
#             dfdat = df[(df["workload"]==workload)&(df["rdelay"]==rdelay)&(df["algorithm"]==alg)]
#             ax.plot(dfdat["bw"],dfdat["time"],label=labels[alg],marker=markers[alg],c=colors[alg],lw=2,markersize=20)
#             ax.xaxis.grid(True,ls='--')
#             ax.yaxis.grid(True,ls='--')


for workload in workloads:
    for alg in algs:
        if alg == "vermilion":
            continue
        # if workload != "deepseek":
        #     continue
        arr = np.zeros((len(bandwidths),len(rdelays)))
        dfalg = df[(df["workload"]==workload)&(df["algorithm"]==alg)]
        dfverm = df[(df["workload"]==workload)&(df["algorithm"]=="vermilion")]
        for j in range(len(rdelays)):
            for i in range(len(bandwidths)):
                algtime = list(dfalg[(dfalg["rdelay"]==rdelays[j])&(dfalg["bw"]==bandwidths[i])]["time"])[0]
                vermtime = list(dfverm[(dfverm["rdelay"]==rdelays[j])&(dfverm["bw"]==bandwidths[i])]["time"])[0]
                arr[i][j]=algtime/vermtime
                print(algtime,vermtime,alg,workload, rdelays[j], bandwidths[i])
        
        print("###############")
        print(workload, alg)
        for row in arr:
            print(" ".join(f"{x:.4f}" for x in row))
        print("")
        
        fig,ax = plt.subplots(1,1, figsize=(8,7))
        
        maxRatio = np.max(arr)
        minRatio = np.min(arr)
        
        modMin = 1/maxRatio
        modMax = max(maxRatio, 1/minRatio)
        
        norm = mcolors.LogNorm(vmin=modMin, vmax=modMax)
        ax = sns.heatmap(arr,norm=norm, cmap="vlag_r", annot=True,fmt=".1f",annot_kws={"size": 14})

        ax.set_xticklabels(rdelaylabels,rotation=35)
        ax.set_yticklabels(bandwidthlabels, rotation=30)
        ax.set_xlabel("Reconfiguration delay")
        ax.set_ylabel("Bandwidth")
        
        fig.tight_layout()
        # ax.set_title(str(workload) + " " + str(alg))
        
        fig.savefig("plots/"+str(workload)+"-"+str(alg)+"-"+"heatmap.pdf")