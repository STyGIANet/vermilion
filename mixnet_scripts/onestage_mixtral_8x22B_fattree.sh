#declare -a bw=(40000 100000 200000 400000 800000)
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
        for workload in "${workloads[@]}"; do
            mkdir -p ./logs/mixtral8x22B_onestage_${mb}_fattree_${b}_${workload}/

            ./htsim_tcp_fattree -flowfile ${new_fbuf_dir}/mixtral8x22B_onestage_dp2_tp8_pp1_ep8_${mb}.fbuf -simtime 36000.1  -speed $((b*1000)) -nodes 512 -ofile nwsim_linkft_${b}_${workload}.txt -ssthresh 10000 -rttnet 1000 -rttrack 1000 -q 10000 -weightmatrix ../../../test/${workload}.txt -logdir ./logs/mixtral8x22B_onestage_${mb}_fattree_${b}_${workload}/ > ./logs/mixtral8x22B_onestage_${mb}_fattree_${b}_${workload}/output.log 2>&1 &
        done
    done
done
