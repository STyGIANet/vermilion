#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(1 10 50 100 500 1000 2000 3000 40000 5000)
declare -a workloads=("num_global_tokens_per_expert")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname $SCRIPT_DIR)"
LOG_DIR="$REPO_DIR/logs"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"

DEBUG=5
COMPILE=0

ARG="$1"
ALGORITHMS=("vermilion" "mixnet" "birkoff" "rotor")

declare -a algorithms_to_run

if [[ "$ARG" == "all" ]]; then
    algorithms_to_run=("${ALGORITHMS[@]}")
else 
    match=0
    for a in "${ALGORITHMS[@]}"; do
        [[ "$a" == "$ARG" ]] && { match=1; break; }
    done
    if [[ $match -eq 1 ]]; then
        algorithms_to_run=("$ARG")
    fi
fi

analyze_timings() {
    local FOLDER="${1:-.}"
    local LOGFILE="$FOLDER/output_err.txt"
    local OUTFILE="$FOLDER/timing_analysis.txt"

    if [[ ! -f "$LOGFILE" ]]; then
        echo "ERROR: $LOGFILE not found." | tee -a "$OUTFILE"
        return 1
    fi

    {
        echo "===== AGGREGATE timings in $FOLDER ====="
        grep "^aggregate.*timing" "$LOGFILE" | \
             awk '{printf "  task_id=%-20s layer=%s region=%s micro_batch=%s  start=%-15s finish=%-15s  duration=%s\n", $5, $7, $9, $11, $13, $15, $17}'
        echo ""
        echo "===== GROUP BY (forward) timings in $FOLDER ====="
        grep "^group by.*timing" "$LOGFILE" | \
        
         awk '{printf "  task_id=%-20s layer=%s region=%s micro_batch=%s  start=%-15s finish=%-15s  duration=%s\n", $6, $8, $10, $12, $14, $16, $18}'


        echo ""
        echo "===== COUNT ====="
        agg_count=$(grep -c "^aggregate.*timing" "$LOGFILE" 2>/dev/null || echo 0)
        grp_count=$(grep -c "^group by.*timing"  "$LOGFILE" 2>/dev/null || echo 0)
        echo "  aggregate : $agg_count lines"
        echo "  group by  : $grp_count lines"
    } | tee "$OUTFILE"

    echo "  [timing analysis saved to $OUTFILE]"
}



summarize_rdelay() {
    local BASE_DIR="$1"
    local MB="$2"
    local BW="$3"
    local ALG="$4"
    local OUTFILE="$BASE_DIR/summary_mb${MB}_bw${BW}_${ALG}.txt"

    echo "===== RDELAY SUMMARY: mb=$MB bw=$BW algorithm=$ALG =====" | tee "$OUTFILE"
    echo "" | tee -a "$OUTFILE"

    for workload in "${workloads[@]}"; do
        for r in "${rdelay[@]}"; do
            local RUN_DIR="$BASE_DIR/mixtral8x22B_${MB}_mixnet_${BW}_${workload}_${r}_${ALG}"
            local ANALYSIS="$RUN_DIR/timing_analysis.txt"

            if [[ ! -f "$ANALYSIS" ]]; then
                echo "  [MISSING] rdelay=$r workload=$workload" | tee -a "$OUTFILE"
                continue
            fi

            echo "---- rdelay=$r workload=$workload ----" | tee -a "$OUTFILE"
            cat "$ANALYSIS" | tee -a "$OUTFILE"
            echo "" | tee -a "$OUTFILE"
        done
    done

    echo "  [summary saved to $OUTFILE]"
}



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
                for workload in "${workloads[@]}"; do
                    DUMP_DIR=$LOG_DIR/vermilion-rdelay/mixtral8x22B_${mb}_mixnet_${b}_${workload}_${r}_${algorithm}
                    # Note: r and periodic_interval are both in nanoseconds units
                    periodic_interval=$(python3 -c "print(max([$r * 10, 800]))")
                    mkdir -p $DUMP_DIR
                    cd "$ALG_DIR"

                    if [[ $algorithm == "vermilion" ]]; then
                        echo "Running Vermilion with periodic reconf interval $periodic_interval ns and rdelay $r ns"
                        ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_ns $periodic_interval  -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt
                    elif [[ $algorithm == "mixnet" ]]; then
                        echo "Running MixNet with rdelay $r ns"
                        ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt 
                     fi
                    # Run timing analysis and save results to $DUMP_DIR
                    analyze_timings "$DUMP_DIR"

                done
            done
            summarize_rdelay "$LOG_DIR/vermilion-rdelay" "$mb" "$b" "$algorithm"
        done
    done
done