# Vermilion

This repository contains the code and scripts required to reproduce the simulation results presented in the **Vermilion** paper.

The repository includes:

* The Vermilion network simulator
* Workload traces used in the experiments
* Scripts for reproducing paper results
* Scripts for running parameter sweeps
* Analysis and plotting utilities

---

## Requirements

Before building and running the experiments, make sure the following dependencies are installed:

* FlatBuffers library
* Python 3
* Pandas
* Seaborn
* Matplotlib

Python dependencies can be installed using:

```bash
pip install pandas seaborn matplotlib
```

---

## Repository Setup

Clone the repository along with its submodules:

```bash
git clone --recurse-submodules git@github.com:STyGIANet/vermilion.git
cd vermilion
```

Extract the workload traces:

```bash
unzip mixnet_flatbuf.zip
```

After extraction, the repository should contain the `mixnet_flatbuf/` directory with the workload files used by the simulator.

---

## Building Vermilion

From the root of the repository, run:

```bash
./build_vermilion_sim
```

The build script performs the required cleanup and compilation steps for the simulator. Internally, it runs commands equivalent to:

```bash
cd src/clos/datacenter
make clean

cd ../
make clean
make -j

cd datacenter/
make -j
```

After a successful build, the simulator binary should be available at:

```text
src/clos/datacenter/htsim_tcp_mixnet
```

---

## Running Vermilion

### 1. Create a directory for logs

From the repository root:

```bash
mkdir logs/test
```

### 2. Move to the simulator directory

```bash
cd src/clos/datacenter
```

### 3. Run the simulator

An example Vermilion simulation command is:

```bash
./htsim_tcp_mixnet \
    -algorithm vermilion \
    -debug 5 \
    -simtime 360000.1 \
    -flowfile ../../../mixnet_flatbuf/mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf \
    -speed 1600000 \
    -alpha 4 \
    -ocs_file ../../../logs/test/nwsim_ocs.txt \
    -ecs_file ../../../logs/test/nwsim_ecs.txt \
    -nodes 1024 \
    -ssthresh 10000 \
    -rtt 1 \
    -q 100000000000 \
    -dp_degree 2 \
    -tp_degree 8 \
    -pp_degree 8 \
    -ep_degree 8 \
    -periodic_reconf 1 \
    -periodic_interval_ns 800 \
    -rdelay 10 \
    -elecspeed 400000 \
    -weightmatrix ../../../test/mixtral_2.txt \
    -logdir ../../../logs/test/ \
    2> ../../../logs/test/output_err.txt \
    > ../../../logs/test/output.txt
```

The simulator output is written to:

```text
logs/test/output.txt
```

and error/debug output is written to:

```text
logs/test/output_err.txt
```

Additional simulator-generated files such as completion time of flows in ocs and ecs are also stored under:

```text
logs/test/
```

---

# Running Experiments

Experiment scripts are located in:

```bash
cd vermilion-scripts
```

Before generating plots, create the plot directory if it does not already exist:

```bash
mkdir -p plots
```

---

## Reproducing Figure 5

Run:

```bash
./figure5.sh
```

This script reproduces **Figure 5** from the paper.

It evaluates the following network designs:

* Vermilion
* Birkhoff
* RotorNet-VLB
* MixNet

The experiments use the **DeepSeek-MoE** workload while varying parameters such as:

* Network bandwidth
* Optical circuit switch reconfiguration delay

The processed results and plots are stored under:

```text
vermilion-scripts/plots/
```

---

To run with different workload:

```bash
./figure5.sh <workload>
```
---
## Running the Full Experiment Suite

Run:

```bash
./vermilion-experiments.sh
```

This script runs experiments for:

* Vermilion
* MixNet
* Birkhoff
* RotorNet

across different configurations, including:

* Reconfiguration delay
* Network bandwidth
* Microbatch size
* Workload

Depending on the parameter sweep, these experiments may take significantly longer than a single simulator run.

---

## Processing Experiment Results

After running the experiments, use:

```bash
./vermilion-results.sh
```

This script processes the raw simulator outputs and converts them into a more convenient format for analysis and plotting.

Run this script **after** `vermilion-experiments.sh` has generated the experiment data.

---

## Plotting Results

To plot results from the processed experiments, run:

```bash
python3 plot_diff_workload.py
```

Make sure the plot directory exists before running the script:

```bash
mkdir plots
```

Generated plots are stored under the `plots/` directory.

---

# Typical Workflow

For a fresh checkout, the complete workflow is:

```bash
# Clone repository
git clone --recurse-submodules git@github.com:STyGIANet/vermilion.git
cd vermilion

# Extract workload traces
unzip mixnet_flatbuf.zip

# Build simulator
./build_vermilion_sim

# Move to experiment scripts
cd vermilion-scripts

# Create plot directory
mkdir -p plots

# Reproduce Figure 5
./figure5.sh
```

For running and analyzing the broader experiment suite:

```bash
./vermilion-experiments.sh
./vermilion-results.sh
python3 plot_diff_workload.py
```

---



## Quick introduction (topology reconfiguration flow)

Starting point
- The first reconfiguration is initiated from `FFTask::cleanup` when the next task in `task_buf` is an all-to-all.

RegionalTopoManager::doNext()
- This method is invoked for topology-related events. There are three relevant status states:
    1. TOPO_LIVE
    2. TOPO_RECONF
    3. TOPO_WAIT_DRAIN (added to explicitly handle drain checks)

Behavior by state
- TOPO_LIVE
    - Indicates an active topology is present.
    - The event handler pauses TCP flows, queues and pipes.
    - If there are no packets in flight or queued, it schedules a `_reconfig` event (same RegionalTopoManager type) after `reconfiguration_delay` and sets the status to `TOPO_RECONF`.

- TOPO_RECONF
    - Signifies that the reconfiguration delay has elapsed.
    - The manager applies the new topology and updates link bandwidth/datarate according to the `conn[][]` matrix.

- TOPO_WAIT_DRAIN
    - Used to explicitly verify that all in-flight/queued packets have drained before proceeding with reconfiguration.

Notes 
- Use the log files under `vermilion/scr/clos/datacentre/logs/` to trace event timings and state transitions.


## Generating an example heatmap

Figure 4 in the paper shows an example with heatmap representation using DLRM data parallel traffic matrix. This example can be reproduced as follows: 

```bash
cd throughput-analysis
python3 example-vermilion.py
```
The following plots will be generated in `plots/` directory

- dlrm-original.pdf (Figure 4a)
- dlrm-normalized.pdf  (Figure 4d)
- dlrm-upscaled.pdf (Figure 4e)
- dlrm-rounded.pdf (Figure 4f)
- dlrm-completeaug.pdf (Figure 4g)
- configuration-model-graph.pdf 
- dlrm-configurationaug.pdf (Figure 4h)
- dlrm-emulated.pdf (Figure 4c)
- dlrm-emulated-oblivious.pdf (Figure 4b)


## Throughput comparison

Figure 7 in the paper compares the throughput of different approaches across various traffic matrices. This can be reproduced as follows:

```bash
cd throughput-analysis
python3 paperCode.py > nsdi2027-throughput-results.txt
cat nsdi2027-throughput-results.txt | grep mygrep > nsdi2027-throughput-results.csv
python3 throughput-plots.py
```

The following plots will be generated in `plots/` directory

- throughput-16.pdf (Figure 7)
- throughput-32.pdf 
- throughput-48.pdf (Figure 11)
- throughput-k.pdf (Figure 8a)
- throughput-nodes.pdf (Figure 8b)


## Opera paths

Figure 6b can be generated using `python plot-opera-paths.py`, where the input file has been generated using [opera-sim](https://github.com/TritonNetworking/opera-sim).