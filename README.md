# MixNet-htsim simulator

This module contains the MixNet-htsim packet-level network simulator. It models the cluster that runs MoE training job with different interconnects. The source code was extended from the TopoOpt simulator from NSDI 2023 and Opera simulator from NSDI 2020, please check the original README file [here](OPERA_README.md).

## Compilation:
To build the MixNet-htsim simulator, from the top level directory run:
```bash
cd src/clos
make
cd datacenter
make
```
We also provide convinent compile scripts [here](./mixnet_scripts/compile.sh)

## Executables

The executables are found in the `src/clos/datacenter` folder. They have the name "htsim_...". The following table provides details on each executable:

| Executable | Network Topology |
|------------|------------------|
| `htsim_tcp_fattree`       | Fat-Tree network topology, single job |
| `htsim_tcp_os_fattree`    | Oversubscribed Fat-Tree where the ToR switches are oversubscribed |
| `htsim_tcp_mixnet`        | Runtime reconfigurable optical-electrical fabric for distributed Mixture-of-Experts training |

## Brief description on source code

MixNet-htsim's major extension from the htsim simulator allows it to take a taskgraph (in FlatBuffer) generated from the FlexFlow DNN training simulator. To achieve this, `src/clos/ffapp.*` was implemented as an API to read and process these such taskgraphs. In addition, a few network topologies are added, notably the dynamic network executable that simulates SiP-ML. The mixnet topology logic can be found in `src/clos/datacenter/mixnet.*` and the regional reconfiguration logic for mixnet is implemented in `src/clos/mixnet_topomanager.*`.

Each topology's "main" function can be found in `src/clos/datacenter/main_tcp_*.cpp`, which provides detailed description on the input arguments for the executable. 

