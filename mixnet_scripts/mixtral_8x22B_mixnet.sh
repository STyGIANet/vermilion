#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(1)
declare -a workloads=("num_global_tokens_per_expert")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname $SCRIPT_DIR)"
LOG_DIR="$REPO_DIR/logs"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"

DEBUG=2
COMPILE=1


N=0

for algorithm in "vermilion" "mixnet";do
   
    
    if [[ $algorithm == "vermilion" ]];then
        ALG_DIR="$REPO_DIR/src/clos/datacenter"
        if [[ $COMPILE -eq 1 ]]; then
            echo "##### Compiling $algorithm ######"
            cd $SCRIPT_DIR
            ./compile.sh > $LOG_DIR/vermilion_compile_output.txt 2> $LOG_DIR/vermilion_compile_error.txt
            echo "##### Compiled Vermilion ######"
        fi
    elif [[ $algorithm == "mixnet" ]];then
        ALG_DIR="$REPO_DIR/mixnet-htsim/src/clos/datacenter"
         if [[ $COMPILE -eq 1 ]]; then
            echo "##### Compiling $algorithm ######"
            cd $REPO_DIR/mixnet-htsim/mixnet_scripts/
            ./compile.sh > $LOG_DIR/mixnet_compile_output.txt 2> $LOG_DIR/mixnet_compile_error.txt
            echo "##### Compiled MixNet ######"
        fi
    fi
    

    for mb in "${microbatchsize[@]}"; do
        for b in "${bw[@]}"; do
            for r in "${rdelay[@]}"; do
                for workload in "${workloads[@]}"; do
                    DUMP_DIR=$LOG_DIR/mixtral8x22B_${mb}_mixnet_${b}_${workload}_${r}_${algorithm}
                    # Note: r and periodic_interval are both in nanoseconds units
                    periodic_interval=$(python3 -c "print(max([$r * 10, 800]))")
                    mkdir -p $DUMP_DIR
                    cd "$ALG_DIR"

                    sleep 2
                    while [[ $(ps aux | grep "htsim_tcp_mixnet" | wc -l) -gt 40 ]];do
                        echo "Waiting for cpu cores.... $N-th experiment "
                        sleep 10
                    done
                    
                    if [[ $EXP == 1 ]];then
                        
                         #/htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 100 -q 8 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_us $periodic_interval  -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt 
                        if [[ $algorithm == "vermilion" ]];then
                            echo "Running Vermilion with periodic reconf interval $periodic_interval ns and rdelay $r ns"
                            (time ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 \
                            -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf \
                            -speed $((b*1000)) \
                            -alpha 4 \
                            -ocs_file nwsim_ocs_${b}.txt \
                            -ecs_file nwsim_ecs_${b}.txt  \
                            -nodes 1024 \
                            -ssthresh 10000 \
                            -rtt 1 \
                            -q 100000000000 \
                            -dp_degree 2 \
                            -tp_degree 8 \
                            -pp_degree 8 \
                            -ep_degree 8 \
                            -periodic_reconf 1 \
                            -periodic_interval_ns $periodic_interval  \
                            -rdelay $r \
                            -weightmatrix $REPO_DIR/test/${workload}.txt \
                            -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt; echo "$DUMP_DIR finished...") & 
                        
                        elif [[ $algorithm == "mixnet" ]];then
                            echo "Running MixNet with rdelay $r ns"
                            (time ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 \
                            -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf \
                            -speed $((b*1000)) \
                            -alpha 4 \
                            -ocs_file nwsim_ocs_${b}.txt \
                            -ecs_file nwsim_ecs_${b}.txt  \
                            -nodes 1024 \
                            -ssthresh 10000 \
                            -rtt 1 \
                            -q 100000000000 \
                            -dp_degree 2 \
                            -tp_degree 8 \
                            -pp_degree 8 \
                            -ep_degree 8 \
                            -rdelay $r \
                            -weightmatrix ../../../test/${workload}.txt \
                            -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt ; echo "$DUMP_DIR finished...") &
                         #./htsim_tcp_mixnet -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1000 -q 2 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_us 0  -rdelay $r -weightmatrix ../../../test/${workload}.txt -logdir ./logs/Non-periodic/ > ./logs/Non-periodic/output.txt 2> ./logs/Non-periodic/output_err.txt 
                        fi
                    
                    fi
                    N=$(( $N+1 ))
                done
            done
        done
    done
done

echo "Total $N experiments finished."