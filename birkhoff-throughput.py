#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:56:03 2023

@author: vamsi
"""

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
# from birkhoff import birkhoff_von_neumann_decomposition as decomp

#%%
N = 4
D = np.ones((N, N))
for i in range(N):
    D[i][i]=0
#%%
zipped_pairs = decomp(D)
coefficients, permutations = zip(*zipped_pairs)
#%%

N = 1024000000

d=N

alphas = [(d-1)*i*0.001 for i in range(500)]

throughput=list()
for alpha in alphas:
    # beta1 = max( [alpha/2 , (3*alpha/2 - d/2)] )
    beta1 = alpha/2 + (alpha/2)*(2)
    # beta1 = alpha/2
    # beta2 = max( [(N/2 - alpha/2), (N -3*alpha/2)] )
    # beta2 = (d - alpha) - (d-alpha)*alpha/N
    # if (beta2<0):
        # print("beta",beta2)
    beta2 = (N-alpha)*(N-alpha)/N + (N -alpha - (N-alpha)*(N-alpha)/N)* 2*np.log(N)/np.log(N-alpha)
    # beta2 = (N-2*alpha) * 1 + (alpha)*(np.ceil(np.log(N)/np.log(N-alpha))+1)
    # beta2 = ((N - alpha) - alpha)/N
    # if (alpha/d==0.5):
        # print((beta1+beta2)/d, beta1/d, beta2/d)
    
    th = N/(beta1 + beta2 )
    throughput.append(th)
    # if(th<0):
# .        print(alpha,beta1,beta2,d-(beta1+beta2),np.log(N)/np.log(d-alpha))

fig,ax=plt.subplots(1,1)
ax.plot([i/d for i in alphas],throughput)
print("throughput",min(throughput))



#%%

