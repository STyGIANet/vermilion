# Overview

This directory contains the source code for the paper (under submission) `Vermilion, Pt.2: Tradeoffs between Throughput and Prediction Accuracy in Reconfigurable Optical Interconnects`.

The code is organized as follows:

- `vermilion2.py`: The main python file to run the experiments. This file contains the code to generate graphs for generalized vermilion and its randomized version, across demand matrices and for different noise levels for the estimated demand matrix. This code then computes the throughput based on the topology constructed for each algorithm (this requires gurobipy and a [license](https://portal.gurobi.com/iam/licenses/list)).

- `Throughput_as_Function.py` contains the source code for computing the throughput. A user does not need to modify or run this file. The `thetaEdgeFormulation` function in this file returns the throughput for a given topology and demand matrix.

- `run.sh` is a one-shot script to run the experiments in parallel. It runs 10 instances of the vermilion2.py script, each with a different noise level parameter.

- `supercomputing-plots.py` contains the code to generate the plots presented in the paper.

- `matrices/` directory contains `.mat` text files that represent the demand matrices used in the experiments.

# Reproducing the Results

The code is written in python 3.11.11 and tested on Linux Fedora 41.

- install dependencies

```bash
pip install -r requirements.txt
```

- Run the experiments in parallel
```bash
chmod +x ./run.sh
./run.sh
```

- Aggregate the results to a single file

```bash
for i in {0..9}; do cat results/sigmetrics-throughput-results-16-$i.csv >> results/sigmetrics-throughput-results-16.csv; done
```

- Plot the results 

```bash
python3 ./supercomputing-plots.py
```