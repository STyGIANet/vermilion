declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a nnodes=(128)
declare -a rdelay=(25)
declare -a workloads=("num_global_tokens_per_expert")
dir="/usr/wkspace/mixnet-sim/mixnet-htsim/src/clos/datacenter"
new_fbuf_dir="/usr/wkspace/mixnet-sim/mixnet-flexflow/results"
cd "$dir" || exit 1
for mb in "${microbatchsize[@]}"; do
    for b in "${bw[@]}"; do
        for r in "${rdelay[@]}"; do
            for workload in "${workloads[@]}"; do
                mkdir -p ./logs/mixtral8x22B_onestage_${mb}_mixnet_${b}_${workload}/
                ./htsim_tcp_mixnet -simtime 3600.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_onestage_dp2_tp8_pp1_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 512 -ssthresh 10000 -rtt 1000 -q 10000 -dp_degree 2 -tp_degree 8 -pp_degree 1 -ep_degree 8 -rdelay $r -weightmatrix ../../../test/${workload}.txt -logdir ./logs/mixtral8x22B_onestage_${mb}_mixnet_${b}_${workload}/ > ./logs/mixtral8x22B_onestage_${mb}_mixnet_${b}_${workload}/output.log 2>&1 &
            done
        done
    done
done
