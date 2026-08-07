#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(800 1600 2400 3200 4800 6400 8000 9600)
declare -a microbatchsize=(8)
declare -a rdelay=(10 100 500 1000 10000 100000 1000000 10000000)
ALLWORKLOADS=("num_global_tokens_per_expert" "mixtral" "deepseek" "mixtral_2" "Phi-3.5-MoE-instruct-gsm8k" "Phi-3.5-MoE-instruct-mmlu" "qwen" "qwen_128")
declare -a workloads=("deepseek")

ALGS=("birkhoff" "vermilion" "rotor" "mixnet")
#ALGS=("birkhoff")

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
ARG="$1"

if [[ -z "$ARG" ]]; then
    workloads=("deepseek")
    echo "No workload argument given, defaulting to: deepseek"
elif [[ "$ARG" == "all" ]]; then
    workloads=("${ALLWORKLOADS[@]}")
else
    match=0
    for a in "${ALLWORKLOADS[@]}"; do
        [[ "$a" == "$ARG" ]] && { match=1; break; }
    done
    if [[ $match -eq 1 ]]; then
        workloads=("$ARG")
    else
        echo "ERROR: unknown workload '$ARG'. Choose one of: ${ALLWORKLOADS[*]} or 'all'"
        exit 1
    fi
fi
VERMILION_COMPILE=0
for algorithm in ${ALGS[@]};do
   
    
    if [[  $algorithm == "vermilion" || $algorithm == "rotor" || $algorithm == "birkhoff" ]];then
        ALG_DIR="$REPO_DIR/src/clos/datacenter"
        if [[ $COMPILE -eq 1 && $VERMILION_COMPILE -eq 0 ]]; then
            echo "##### Compiling $algorithm ######"
            cd $SCRIPT_DIR
            ./compile.sh > $LOG_DIR/${algorithm}_compile_output.txt 2> $LOG_DIR/${algorithm}_compile_error.txt
            echo "##### Compiled $algorithm ######"
            VERMILION_COMPILE=1
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
                    DUMP_DIR=$LOG_DIR/workload_${mb}_${b}_${workload}_${r}_${algorithm}
                    # Note: r and periodic_interval are both in nanoseconds units
                    periodic_interval=$(python3 -c "print(max([$r * 10, 800]))")
                    # periodic_interval=$(python3 -c "print(max([45000]))")
                    mkdir -p $DUMP_DIR
                    cd "$ALG_DIR"

                    while [[ $(ps aux | grep "htsim_tcp_mixnet" | wc -l) -gt $NCORE ]];do
                        echo "Waiting for cpu cores.... $N-th experiment "
                        sleep 2
                    done
                    
                    
                    echo "Running $algorithm with rdelay $r ns, bw $b workload $workload experiment Id $N"
                    if [[ $EXP == 1 ]];then
                        if [[ $algorithm == "vermilion" || $algorithm == "rotor" || $algorithm == "birkhoff" ]];then
                            time ( 
                            ./htsim_tcp_mixnet -algorithm $algorithm -debug $DEBUG -simtime 360000.1 \
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
                            -elecspeed $((ELEC_BW*1000)) \
                            -weightmatrix $REPO_DIR/test/${workload}.txt \
                            -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt; echo "$DUMP_DIR finished...")&
                        
                        elif [[ $algorithm == "mixnet" ]];then
                            time (./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 \
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
                            -elecspeed $((ELEC_BW*1000)) \
                            -weightmatrix $REPO_DIR/test/${workload}.txt \
                            -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt ; echo "$DUMP_DIR finished...")&
                        fi
                    
                    fi
                    N=$(( $N+1 ))
                done
            done
        done
    done
done
while [[ $(ps aux | grep "htsim_tcp_mixnet" | wc -l) -gt $NCORE ]];do
    echo "Waiting for experiments to finish.... Total $N "
    sleep 10
done

echo "Waiting for all experiments to finish..."
wait
echo "All experiments finished."

##Analyse the results
cd $SCRIPT_DIR
NCORE=100
N=0
echo "algorithm bw rdelay workload time" > "results.txt"
for algorithm in ${ALGS[@]};do
    for mb in "${microbatchsize[@]}"; do
        for b in "${bw[@]}"; do
            for r in "${rdelay[@]}"; do
                for workload in "${workloads[@]}"; do
                    DUMP_DIR=$LOG_DIR/workload_${mb}_${b}_${workload}_${r}_${algorithm}
                    # echo $DUMP_DIR
                    # time in milliseconds
                    # Note: 0 is an error, not magically fast
                    COMPLETION_TIME=$(awk '/duration/ {sum += $NF; n++} END {if (n>0) printf "%.4f\n", sum/n/1e9; else printf "0"}' "$DUMP_DIR/output_err.txt")
                    echo "$algorithm $b $r $workload $COMPLETION_TIME" >> "results.txt"
                    N=$(( $N+1 ))
                done
            done
        done
    done
done


echo "Waiting for all experiments to finish..."
wait
echo "All experiments finished."
echo "Total $N experiments finished."

#Plot the result
if ! python3 "$SCRIPT_DIR/plot_diff_workload.py"; then
    echo "ERROR: plotting script failed"
fi

echo "Total $N experiments finished."