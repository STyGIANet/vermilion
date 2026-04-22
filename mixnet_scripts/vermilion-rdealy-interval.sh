#!/bin/bash
# RTT is in nanoseconds, periodic_interval_ns and rdelay are also in nanoseconds units
declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(1 10 50 100 500 1000)
declare -a interval=(800 1600 2400 3000 40000 50000 60000 70000 80000 10000 15000)
declare -a workloads=("num_global_tokens_per_expert")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname $SCRIPT_DIR)"
LOG_DIR="$REPO_DIR/logs"
new_fbuf_dir="$REPO_DIR/mixnet_flatbuf"

DEBUG=5
COMPILE=0


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

    echo "===== INTERVAL SUMMARY: mb=$MB bw=$BW algorithm=$ALG =====" | tee "$OUTFILE"
    echo "" | tee -a "$OUTFILE"

    for workload in "${workloads[@]}"; do
        for r in "${rdelay[@]}"; do
            for i in "${interval[@]}"; do
                local RUN_DIR="$BASE_DIR/mixtral8x22B_${MB}_mixnet_${BW}_${workload}_${r}_${i}_${ALG}"
                local ANALYSIS="$RUN_DIR/timing_analysis.txt"

                if [[ ! -f "$ANALYSIS" ]]; then
                    echo "  [MISSING] rdelay=$r interval=$i workload=$workload" | tee -a "$OUTFILE"
                    continue
                fi

                echo "---- rdelay=$r interval=$i workload=$workload ----" | tee -a "$OUTFILE"
                cat "$ANALYSIS" | tee -a "$OUTFILE"
                echo "" | tee -a "$OUTFILE"
            done
        done
    done

    echo "  [summary saved to $OUTFILE]"
}


for algorithm in "vermilion" "mixnet"; do

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
                for i in "${interval[@]}"; do
                    for workload in "${workloads[@]}"; do
                        DUMP_DIR=$LOG_DIR/vermilion-rdelay-interval/mixtral8x22B_${mb}_mixnet_${b}_${workload}_${r}_${i}_${algorithm}
                        mkdir -p $DUMP_DIR
                        cd "$ALG_DIR"

                        # if [[ $algorithm == "vermilion" ]]; then
                        #     echo "Running Vermilion with periodic reconf interval $i ns and rdelay $r ns"
                        #     ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_ns $i -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt
                        # elif [[ $algorithm == "mixnet" ]]; then
                        #     echo "Running MixNet with rdelay $r ns"
                        #     ./htsim_tcp_mixnet -debug $DEBUG -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -alpha 4 -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt -nodes 1024 -ssthresh 10000 -rtt 1 -q 100000000000 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -rdelay $r -weightmatrix $REPO_DIR/test/${workload}.txt -logdir $DUMP_DIR > $DUMP_DIR/output.txt 2> $DUMP_DIR/output_err.txt
                        # fi

                        analyze_timings "$DUMP_DIR"
                    done
                done
            done
            summarize_rdelay "$LOG_DIR/vermilion-rdelay-interval" "$mb" "$b" "$algorithm"
        done
    done
done