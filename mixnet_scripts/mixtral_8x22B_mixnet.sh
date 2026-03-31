declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(0.05)
declare -a workloads=("num_global_tokens_per_expert")
dir="./../src/clos/datacenter"
new_fbuf_dir="./../../../mixnet_flatbuf/"
cd "$dir" || exit 1
for mb in "${microbatchsize[@]}"; do
    for b in "${bw[@]}"; do
        for r in "${rdelay[@]}"; do
            for workload in "${workloads[@]}"; do
                mkdir -p ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/
                ./htsim_tcp_mixnet -simtime 360000.1 -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf -speed $((b*1000)) -ocs_file nwsim_ocs_${b}.txt -ecs_file nwsim_ecs_${b}.txt  -nodes 1024 -ssthresh 10000 -rtt 1000 -q 2 -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 -periodic_reconf 1 -periodic_interval_us 60  -rdelay $r -weightmatrix ../../../test/${workload}.txt -logdir ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/ > ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/output.txt 2> ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/output_err.txt 
            done
        done
    done
done
