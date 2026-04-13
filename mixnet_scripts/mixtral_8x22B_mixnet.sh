#!/bin/bash

declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(0.05)
declare -a workloads=("num_global_tokens_per_expert")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR../")"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"

for algorithm in "vermilion" "mixnet";do
    if [[ $algorithm == "vermilion" ]];then
        ALG_DIR="$REPO_DIR/src/clos/datacenter"
        cd $SCRIPT_DIR
        ./compile.sh
        echo "##### Compiled Vermilion ######"
    elif [[ $algorithm == "mixnet" ]];then
        ALG_DIR="$REPO_DIR/mixnet-htsim/src/clos/datacenter"
        cd $REPO_DIR/mixnet-htsim/mixnet_scripts/
        ./compile.sh
        echo "##### Compiled MixNet ######"
    fi
    
    for mb in "${microbatchsize[@]}"; do
        for b in "${bw[@]}"; do
            for r in "${rdelay[@]}"; do
                for workload in "${workloads[@]}"; do
                    DUMP_DIR=$REPO_DIR/logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}_${r}_${algorithm}
                    mkdir -p $DUMP_DIR
                    $ALG_DIR/htsim_tcp_mixnet -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1 -q 8 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_us 400  -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt 
                    #./htsim_tcp_mixnet -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1000 -q 2 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_us 0  -rdelay $r -weightmatrix ../../../test/${workload}.txt -logdir ./logs/Non-periodic/ > ./logs/Non-periodic/output.txt 2> ./logs/Non-periodic/output_err.txt 
                done
            done
        done
    done
done