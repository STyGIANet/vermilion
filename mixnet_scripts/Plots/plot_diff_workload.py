
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:41:00 2026

@author: deeksha-p-rao
"""

import pandas as pd
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

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

df = load_data("results.txt")
data = build_nested_dict(df)

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
