## Build and run

Prerequisites
- Place the `mixnet_flatbuf` file in the project root (the main folder).

Build
- Run the build script (it performs a simple clean + build):
```bash
./build_vermilion_sim
```

Run
- Change into the scripts folder and run the simulation binary:
```bash
cd vermilion/mixnet_scripts/
./mixtral_8x22B_mixnet
```
- The binary accepts flags to control:
    - reconfiguration delay (-periodic_interval_us 20)
    - periodic reconfiguration flag (-periodic_reconf 1 )
    - timing workload (choose from the options in the  `mixnet_flatbuf` directory) (workloads=("num_global_tokens_per_expert"))

Outputs
- Simulation logs are written to:
```
vermilion/scr/clos/datacentre/logs/<flatbuf_filename>
```
(Replace `<flatbuf_filename>` with the flatbuf file used at runtime.)

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