#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(100 400 800 1600)
declare -a microbatchsize=(8)
declare -a rdelay=(1 10 50 100 500 1000 10000 100000 1000000 10000000)
# declare -a interval=(800 1600 2400 3000 4000 5000 6000 7000 8000 10000)
declare -a workloads=("num_global_tokens_per_expert")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname $SCRIPT_DIR)"
LOG_DIR="$REPO_DIR/logs"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"

DEBUG=5
COMPILE=1

N=0
N_Core=100
EXP=$1
ARG="$1"
ALGORITHMS=("vermilion" "mixnet" "birkoff" "rotor")

declare -a algorithms_to_run

if [[ -z "$ARG" ]]; then
    algorithms_to_run=("vermilion")
    echo "No algorithm argument given, defaulting to: vermilion"
elif [[ "$ARG" == "all" ]]; then
    algorithms_to_run=("${ALGORITHMS[@]}")
else
    match=0
    for a in "${ALGORITHMS[@]}"; do
        [[ "$a" == "$ARG" ]] && { match=1; break; }
    done
    if [[ $match -eq 1 ]]; then
        algorithms_to_run=("$ARG")
    else
        echo "ERROR: unknown algorithm '$ARG'. Choose one of: ${ALGORITHMS[*]} or 'all'"
        exit 1
    fi
fi
for algorithm in "${algorithms_to_run[@]}"; do

    if [[ $algorithm == "vermilion" ]]; then
        ALG_DIR="$REPO_DIR/src/clos/datacenter"
        if [[ $COMPILE -eq 1 ]]; then
            echo "##### Compiling $algorithm ######"
            cd $SCRIPT_DIR
            ./compile.sh > $LOG_DIR/vermilion_compile_output.txt 2> $LOG_DIR/vermilion_compile_error.txt
            echo "##### Compiled Vermilion ######"
        fi
    elif [[ $algorithm == "mixnet" ]]; then
        ALG_DIR="$REPO_DIR/mixnet-htsim/src/clos/datacenter"
        if [[ $COMPILE -eq 1 ]]; then
            echo "##### Compiling $algorithm ######"
            cd $SCRIPT_DIR
            ./compile.sh > $LOG_DIR/mixnet_compile_output.txt 2> $LOG_DIR/mixnet_compile_error.txt
            echo "##### Compiled MixNet ######"
        fi
    fi

    for mb in "${microbatchsize[@]}"; do
        for b in "${bw[@]}"; do
            for r in "${rdelay[@]}"; do
                # for i in "${interval[@]}"; do
                    i=$(python3 -c "print(max([$r*10,800]))")
                    for workload in "${workloads[@]}"; do
                        DUMP_DIR=$LOG_DIR/vermilion-bw-rdelay-interval/mixtral8x22B_${mb}_mixnet_${b}_${workload}_${r}_${i}_${algorithm}
                        mkdir -p $DUMP_DIR
                        cd "$ALG_DIR"

                        sleep 2
                        while [[ $(ps aux | grep "htsim_tcp_mixnet" | wc -l) -gt $N_Core ]];do
                            echo "Waiting for cpu cores.... $N-th experiment "
                            sleep 10
                        done
                        
                        if [[ $EXP == 1 ]];then

                            if [[ $algorithm == "vermilion" ]]; then
                                echo "Running Vermilion with periodic reconf interval $i ns and rdelay $r ns"
                                ./htsim_tcp_mixnet -algorithm $algorithm -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_ns $i -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt
                            elif [[ $algorithm == "mixnet" ]]; then
                                echo "Running MixNet with rdelay $r ns"
                                ./htsim_tcp_mixnet -algorithm $algorithm -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt
                            fi

                            N=$(( $N+1 ))
                            analyze_timings "$DUMP_DIR"
                        fi
                    done
                # done
            done
            # summarize_rdelay "$LOG_DIR/vermilion-bw-rdelay-interval" "$mb" "$b" "$algorithm"
        done
    done
done

echo "Total $N experiments finished."
