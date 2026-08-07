#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:33:39 2024

@author: vamsi
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plotsdir = "/home/vamsi/src/phd/writings/rdcn-throughput/master/plots/"
directory="/home/vamsi/src/phd/codebase/rdcn-throughput/"

#%%

plt.rcParams.update({'font.size': 14})
# Load the updated dataset
df_updated = pd.read_csv(directory+'dump/throughput.csv')
###################################################
# First, filter entries for both 'demand-aware-static' and 'static' networkTypes
demand_aware_static = df_updated[df_updated['networkType'] == 'demand-aware-static']
static = df_updated[df_updated['networkType'] == 'static']

# Prepare to update the throughput values for 'demand-aware-static' with the maximum between
# 'demand-aware-static' and 'static' for the corresponding degree and matrix
# This involves merging the two filtered dataframes on 'degree' and 'matrix', then comparing throughput values

# Merge on 'degree' and 'matrix'
merged = pd.merge(demand_aware_static, static, on=['degree', 'matrix'], suffixes=('_das', '_static'))

# Calculate the maximum throughput value between 'demand-aware-static' and 'static' for each pair
merged['throughput_max'] = merged[['throughput_das', 'throughput_static']].max(axis=1)

# Update the 'demand-aware-static' entries in the original dataframe with these maximum throughput values
for index, row in merged.iterrows():
    df_updated.loc[(df_updated['networkType'] == 'demand-aware-static') & (df_updated['degree'] == row['degree']) & (df_updated['matrix'] == row['matrix']), 'throughput'] = row['throughput_max']
###################################################

for degree in [16, 14, 12, 10, 8, 6, 4]:
    
    # Filter the dataset for entries where degree is 4
    df_updated_degree = df_updated[df_updated['degree'] == degree]
    
    # Mapping of matrix names to updated names for clarity
    matrices = [
        "chessboard-16", "uniform-16", "permutation-16", "skew-16-0.0", "skew-16-0.1", "skew-16-0.2",
        "skew-16-0.3", "skew-16-0.4", "skew-16-0.5", "skew-16-0.6", "skew-16-0.7", "skew-16-0.8",
        "skew-16-0.9", "skew-16-1.0", "data-parallelism", "hybrid-parallelism", "heatmap2", "heatmap3"
    ]
    updated_names = [
        "Chessboard", "Uniform", "Permutation", "U+P 0", "U+P 0.1", "U+P 0.2", "U+P 0.3", "U+P 0.4",
        "U+P 0.5", "U+P 0.6", "U+P 0.7", "U+P 0.8", "U+P 0.9", "U+P 1.0", "Data parallelism",
        "Hybrid parallelism", "DLRM +3 perm", "DLRM +7 perm"
    ]
    
    
    ######
    matrix_mapping = dict(zip(matrices, updated_names))
    df_updated_degree['matrix'] = df_updated_degree['matrix'].map(matrix_mapping)
    # excluding U+P=0, this is same as uniform
    df_updated_degree = df_updated_degree[df_updated_degree['matrix'] != 'U+P 0']
    # excluding U+P=1, this is same as permutation
    df_updated_degree = df_updated_degree[df_updated_degree['matrix'] != 'U+P 1.0']
    
    custom_order = [
        "Chessboard", "Uniform", "U+P 0.1", "U+P 0.2", "U+P 0.3", "U+P 0.4",
        "U+P 0.5", "U+P 0.6", "U+P 0.7", "U+P 0.8", "U+P 0.9", "Permutation", "Data parallelism",
        "Hybrid parallelism", "DLRM +3 perm", "DLRM +7 perm"
    ]
    
    # Ensure the dataset is ordered according to the custom order
    # This step involves mapping the custom order to an orderable list (like integers) that pandas can sort by
    order_mapping = {matrix: i for i, matrix in enumerate(custom_order)}
    df_updated_degree['order'] = df_updated_degree['matrix'].map(order_mapping)
    
    # Now sort by this order
    df_updated_degree = df_updated_degree.sort_values(by='order')
    
    # Unique matrices and network types for plotting
    unique_matrices = df_updated_degree['matrix'].unique()
    unique_network_types = ["static","demand-aware-static","oblivious","demand-aware-periodic"]
    network_type_labels={}
    network_type_labels["static"]="Static (not reconfigurable)"
    network_type_labels["demand-aware-static"]="Demand-aware static (one-shot reconfigurable)"
    network_type_labels["oblivious"]="Demand-oblivious (reconfigurable)"
    network_type_labels["demand-aware-periodic"]="Demand-aware periodic (reconfigurable)"
    
    # Manual color selection for bars
    colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
    # Hatches for further distinction among bars
    hatches = ['/', '\\', '-', 'x', '+', '|', 'o', 'O', '.', '*']
    
    # Setting up the figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Bar width and group settings for spacing
    bar_width = 0.3
    group_width = bar_width * len(unique_network_types) + (bar_width * 0.5)  # Original spacing
    group_spacing = bar_width * 1.5  # Additional spacing between groups
    group_positions = [i * (group_width + group_spacing) for i in range(len(unique_matrices))]
    
    # Plotting bars with adjusted settings
    for i, network_type in enumerate(unique_network_types):
        positions = [x + (bar_width * i) for x in group_positions]
        throughput_values = [
            df_updated_degree[(df_updated_degree['matrix'] == matrix) & (df_updated_degree['networkType'] == network_type)]['throughput'].values[0]
            if df_updated_degree[(df_updated_degree['matrix'] == matrix) & (df_updated_degree['networkType'] == network_type)].shape[0] > 0
            else 0
            for matrix in unique_matrices
        ]
        ax.bar(positions, throughput_values, color=colors[i % len(colors)], width=bar_width, edgecolor='black', label=network_type_labels[network_type], hatch=hatches[i % len(hatches)],alpha=0.6)
        
        ax.scatter(positions[np.argmin(throughput_values)], throughput_values[np.argmin(throughput_values)], marker = "*", s=200, c=colors[i % len(colors)],edgecolors='black')
        indexpos = throughput_values.index(sorted(throughput_values)[1])
        secondmin = sorted(throughput_values)[1]
        ax.scatter(positions[indexpos], secondmin, marker = "*", s=200, c=colors[i % len(colors)],edgecolors='black')
    # Final plot adjustments for aesthetics
    if degree==4 or degree==16:
        ax.set_xlabel('Demand Matrix')
    ax.set_ylabel('Throughput')
    ax.set_ylim(0.2,1.1)
    ax.set_xticks([r + (group_width + group_spacing)/2 - bar_width/2 for r in group_positions])
    ax.set_xticklabels(unique_matrices, rotation=25, ha="right")
    if degree==4 or degree==8:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,framealpha=1)
    ax.yaxis.grid(True, linestyle='--')
    ax.set_xlim([group_positions[0] - bar_width*2, group_positions[-1] + bar_width * 5])
    
    # if degree!=4:
        # ax.text(0.5, 0.85, 'Degree = '+str(degree), ha='center', va='bottom', transform=ax.transAxes)

    
    
    plt.tight_layout()
    plt.show()
    fig.savefig(plotsdir+'degree-'+str(degree)+'.pdf')


#%%

plt.rcParams.update({'font.size': 18})

# Load the dataset
df_updated = pd.read_csv(directory+'dump/throughput.csv')
###################################################
# First, filter entries for both 'demand-aware-static' and 'static' networkTypes
demand_aware_static = df_updated[df_updated['networkType'] == 'demand-aware-static']
static = df_updated[df_updated['networkType'] == 'static']

# Prepare to update the throughput values for 'demand-aware-static' with the maximum between
# 'demand-aware-static' and 'static' for the corresponding degree and matrix
# This involves merging the two filtered dataframes on 'degree' and 'matrix', then comparing throughput values

# Merge on 'degree' and 'matrix'
merged = pd.merge(demand_aware_static, static, on=['degree', 'matrix'], suffixes=('_das', '_static'))

# Calculate the maximum throughput value between 'demand-aware-static' and 'static' for each pair
merged['throughput_max'] = merged[['throughput_das', 'throughput_static']].max(axis=1)

# Update the 'demand-aware-static' entries in the original dataframe with these maximum throughput values
for index, row in merged.iterrows():
    df_updated.loc[(df_updated['networkType'] == 'demand-aware-static') & (df_updated['degree'] == row['degree']) & (df_updated['matrix'] == row['matrix']), 'throughput'] = row['throughput_max']
###################################################

# Group by 'degree' and 'networkType', then calculate the minimum 'throughput' for each group
min_throughput = df_updated.groupby(['degree', 'networkType'])['throughput'].min().reset_index()

# Define distinct markers and colors for the plot
markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'x']
colors = ['#88b7db', '#dbd788', '#c27676', '#7dc276', 'sandybrown', 'lightcoral', 'grey', 'gold']

# Plotting
fig, ax = plt.subplots(figsize=(8,6))

# Unique network types for plotting
network_types = min_throughput['networkType'].unique()
network_types=["static","demand-aware-static","oblivious","demand-aware-periodic"]
network_type_labels={}
network_type_labels["static"]="Static"
network_type_labels["demand-aware-static"]="Demand-aware static"
network_type_labels["oblivious"]="Demand-oblivious"
network_type_labels["demand-aware-periodic"]="Demand-aware periodic"

for i, network_type in enumerate(network_types):
    # Ensure cycling through colors and markers for different network types
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    # Filter data for the current network type
    data = min_throughput[min_throughput['networkType'] == network_type]
    # Plot with customizations
    ax.plot(data['degree'], data['throughput'], marker=marker, markersize=20, linewidth=4, color=color, label=network_type_labels[network_type])

# Final plot adjustments
ax.set_xlabel('Degree')
ax.set_ylabel('Throughput (worst-case)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2,framealpha=0)
ax.xaxis.grid(True,ls='--')
ax.yaxis.grid(True,ls='--')
ax.set_ylim(0.2,1)
y_start = min_throughput[(min_throughput['networkType'] == 'oblivious') & (min_throughput['degree'] == 4)]['throughput'].values[0]
y_end = min_throughput[(min_throughput['networkType'] == 'demand-aware-periodic') & (min_throughput['degree'] == 4)]['throughput'].values[0]-0.1

# Add arrow
ax.annotate('', xy=(14, y_end), xytext=(14, y_start),
            arrowprops=dict(arrowstyle="-|>", color='black',lw=3))

# Add text for "30% improvement"
ax.text(16.5, y_end, '30% improvement', horizontalalignment='right',fontstyle='italic')

plt.show()
fig.tight_layout()
fig.savefig(plotsdir+'min-throughput.pdf')

# df_updated.to_csv(directory+"dump/throughput-updated.csv")

#%%


# Filter the dataset for entries with degree = 4
df_degree_4 = df_updated[df_updated['degree'] == 4]

# Prepare a list to store the improvement results
improvement_results = []

# Iterate over each matrix to calculate the percentage improvement and improvement factor
for matrix in df_degree_4['matrix'].unique():
    # Filter for 'demand-aware-periodic' throughput at the current matrix and degree
    dap_throughput = df_degree_4[(df_degree_4['matrix'] == matrix) & (df_degree_4['networkType'] == 'demand-aware-periodic')]['throughput'].values
    if dap_throughput.size > 0:
        dap_throughput = dap_throughput[0]
        
        # Compare against each other network type
        for network_type in df_degree_4['networkType'].unique():
            if network_type != 'demand-aware-periodic':
                other_throughput = df_degree_4[(df_degree_4['matrix'] == matrix) & (df_degree_4['networkType'] == network_type)]['throughput'].values
                if other_throughput.size > 0:
                    other_throughput = other_throughput[0]
                    # Calculate percentage improvement and improvement factor
                    percentage_improvement = ((dap_throughput - other_throughput) / other_throughput) * 100
                    improvement_factor = 1 + (percentage_improvement / 100)
                    # Store the results
                    improvement_results.append({
                        'Matrix': matrix,
                        'Compared Network Type': network_type,
                        'Percentage Improvement': percentage_improvement,
                        'Improvement Factor': improvement_factor
                    })

# Convert the results list to a DataFrame for easier viewing
df_improvement_results = pd.DataFrame(improvement_results)

# Display the first few rows of the improvement results
# print(df_improvement_results.head())

df_improvement_results.to_csv(directory+'dump/comparisons.csv')
