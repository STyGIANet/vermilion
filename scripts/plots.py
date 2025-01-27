import os
import requests
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
plots_dir = "/home/vamsi/src/phd/writings/rdcn-throughput/sigcomm2025/plots/"

#%%
plt.rcParams.update({'font.size': 14})


fig, ax = plt.subplots(1,1,figsize=(6,5))

df = pd.read_csv("timing-results-cpu.csv", delimiter=' ',usecols=[0,1])

positions_cpu = np.arange(1, 9) - 0.1
positions_gpu = np.arange(1, 9) + 0.1

times = list()
for i in range(8):
    n = 2**(i+1)
    times.append(df[df['nTors'] == n]["time"])
    print(2**(i+1), np.mean(times[-1]))
cpu_boxplot = ax.boxplot(times,positions=positions_cpu, patch_artist=True, boxprops=dict(facecolor='blue'))

df = pd.read_csv("timing-results-gpu.csv", delimiter=' ',usecols=[0,1])

times = list()
for i in range(8):
    n = 2**(i+1)
    times.append(df[df['nTors'] == n]["time"])
    print(2**(i+1), np.mean(times[-1]))
gpu_boxplot = ax.boxplot(times,positions=positions_gpu, patch_artist=True, boxprops=dict(facecolor='orange'))

ax.legend([cpu_boxplot["boxes"][0], gpu_boxplot["boxes"][0]], ["CPU", "GPU"], loc="upper left")


#%%
ax.set_yscale('log')
ax.set_ylim(10**2,10**7)
ax.set_yticks([10**i for i in range(2,8)])
ax.set_yticklabels(["100 ns", "1 "+r'$\mu s$', "10 "+r'$\mu s$', "100 "+r'$\mu s$', "1 ms", "10 ms"])
ax.set_xticks(np.arange(1,9))
ax.set_xticklabels([(2**i) for i in range(1,9)], rotation=30)
ax.set_xlabel("Number of ToR switches")
ax.set_ylabel("Matching computation time")
ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')
fig.tight_layout()
fig.savefig(plots_dir+'matching-times.pdf')
plt.show()