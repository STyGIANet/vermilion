#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(800 1600 2400 3200 4800 6400 8000 9600)
declare -a microbatchsize=(8)
declare -a rdelay=(10 100 500 1000 10000 100000 1000000 10000000)
declare -a workloads=("num_global_tokens_per_expert" "mixtral" "deepseek" "mixtral_2" "Phi-3.5-MoE-instruct-gsm8k" "Phi-3.5-MoE-instruct-mmlu" "qwen" "qwen_128")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname $SCRIPT_DIR)"
LOG_DIR="$REPO_DIR/logs"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"
DEBUG=4
COMPILE=1

EXP=1
N=0

ELEC_BW=400

NCORE=100

for algorithm in "vermilion" "birkhoff";do
    for mb in "${microbatchsize[@]}"; do
        for b in "${bw[@]}"; do
            for r in "${rdelay[@]}"; do
                for workload in "${workloads[@]}"; do
                    DUMP_DIR=$LOG_DIR/workload_${mb}_${b}_${workload}_${r}_${algorithm}
                    # echo $DUMP_DIR
                    # time in milliseconds
                    # Note: 0 is an error, not magically fast
                    COMPLETION_TIME=$(awk '/duration/ {sum += $NF; n++} END {if (n>0) printf "%.4f\n", sum/n/1e9; else printf "0"}' "$DUMP_DIR/output_err.txt")
                    echo "$algorithm $b $r $workload $COMPLETION_TIME"
                    N=$(( $N+1 ))
                done
            done
        done
    done
done

echo "Total $N experiments finished."