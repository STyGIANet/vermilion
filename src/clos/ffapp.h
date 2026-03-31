#ifndef FF_APP_H
#define FF_APP_H

#include "loggers.h"
// #undef max 

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "topology.h"
#include "flat_topology.h"
#include "eventlist.h"
#include "ndp.h"
#include "taskgraph_generated.h"
#include "mixnet_topomanager.h"
// #include "taskgraph.pb.h"
#include "tcp.h"
#define NUM_GPU_PER_NODE 8

/*
 * An application that takes a Flex-flow generated task graph
 * and simulates it on top of the opera network
 */
template<typename T>
std::string vectorToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
bool check_maps_equal(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map1,
                      const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map2);

bool check_symmetric_positions_equal(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map1,
                                     const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map2);
void print_operator_sizes_matrix(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash>& operator_sizes, int matrix_size, int tp_degree, int dp_degree);
class FFApplication;

class FFDevice {
public:
    enum FFDeviceType {
        DEVICE_GPU,
        DEVICE_CPU,
        DEVICE_GPU_COMM,
        DEVICE_DRAM_COMM,
        DEVICE_NW_COMM,
    };

    enum FFDeviceState {
        DEVICE_IDLE,
        DEVICE_BUSY,
    };
    FFApplication * ffapp;

    int node_id, gpu_id; // TODO: check this two value
    float bandwidth;
    FFDeviceType type;
    FFDeviceState state;

    int from_gpu, to_gpu, from_node, to_node;

    simtime_picosec busy_up_to;
    // int nqueued_tasks;

    FFDevice(FFApplication * ffapp, std::string type, float bandwidth, int node_id, int gpu_id, 
             int from_node, int to_node, int from_gpu, int to_gpu);
    FFDevice(FFApplication * ffapp, FlatBufTaskGraph::DeviceType devtype, uint64_t nodeid, 
             uint64_t deviceproperty, uint64_t bandwidth);
};

class FFTask : public EventSource {
public:
    FFApplication * ffapp;
    // static EventList & evl;

    enum FFTaskType {
        TASK_FORWARD,
        TASK_BACKWARD,
        TASK_COMM,
        TASK_UPDATE,
        TASK_BARRIER,
        TASK_NOMINAL_COMM,
        TASK_P2P,
        TASK_SUB_ALLREDUCE,
        TASK_DP_ALLREDUCE,
        TASK_TP_ALLREDUCE,
        TASK_ALLREDUCE,
        TASK_REDUCESCATTER,
        TASK_ALLGATHER,
        TASK_ALLTOALL,
    };

    enum FFTaskState {
        TASK_NOT_READY,
        TASK_READY,
        TASK_RUNNING,
        TASK_FINISHED,
    };

    void add_nextask(FFTask * task);

    void taskstart();
	void cleanup();
	void start_flow();
    
    virtual void doNextEvent(); // call task event
    void execute_compute();
    string get_string_type();
    virtual void updatetrafficmatrix() {} // default implementation does nothing
    virtual void reset() {
        state = FFTask::TASK_NOT_READY;
        ready_time = eventlist().now();
        start_time = 0;
        finish_time = 0;
    }

    FFTaskType type;
    FFTaskState state;
    FFDevice* device;
    int counter;
    uint64_t taskid;
    uint64_t xfersize = 0;
    std::vector<uint64_t> next_tasks;
    int src_node, dst_node;
    std::string name;
	simtime_picosec ready_time, run_time;
	simtime_picosec start_time, finish_time;
    
    std::string info;
    int micro_batch_id;
    int layer_id;
    int target_micro_batch_id;
    int target_layer_id;

    FFTask(FFApplication * ffapp, std::string type, FFDevice * device, uint64_t taskid, uint64_t xfersize, float runtime, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id);
    // FFTask(TaskGraphProtoBuf::Task_SimTaskType tasktype, FFDevice * device, uint64_t xfersize, float runtime);
    FFTask(FFApplication * ffapp, FlatBufTaskGraph::SimTaskType tasktype, FFDevice * device, uint64_t taskid, uint64_t xfersize, float runtime, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id);
    FFTask(FFApplication * ffapp, FFTaskType tasktype, uint64_t taskid, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id);
};

class FFNewRingAllreduce;

struct FFNewRingAllreduceFlow {
    FFNewRingAllreduce * ar;
    int id;
    int src_idx; 
    int ring_idx;
    int round;
};

class FFNewRingAllreduce : public FFTask {

public:
    FFNewRingAllreduce(FFApplication * ffapp, 
    uint64_t taskid, std::vector<uint64_t> ng, 
        const std::vector<std::vector<int>>& jumps, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFNewRingAllreduce() = default;

    std::vector<uint64_t> node_group; // group of nodes in the order of the ring
    const std::vector<std::vector<int>>& jumps;

    uint64_t operator_size;      // total data size of the operator
    std::vector<int> total_jump;
    int finished_rings;

    std::vector<int> finished_curr_round;
    std::vector<int> curr_round;
    std::vector<std::vector<int>> finished_rounds;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        std::fill(finished_curr_round.begin(), finished_curr_round.end(), 0);
        std::fill(curr_round.begin(), curr_round.end(), 0);
        for (auto & v: finished_rounds) {
            std::fill(v.begin(), v.end(), 0);
        }
        finished_rings = 0;
    }

    void start_flow(int src_idx, const std::vector<int>& jump, int ring_id, int id);
};

void ar_finish_newring(void * arinfo);

class FFRingAllreduce;

struct FFRingAllreduceFlow {
    FFRingAllreduce * ar;
    int id;
    int src_idx;
    int round;
};

class FFRingAllreduce : public FFTask {

public:
    FFRingAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFRingAllreduce() = default;

    std::vector<uint64_t> node_group; // group of nodes in the order of the ring
    uint32_t operator_size;      // total data size of the operator
    int finished_partitions;     // number of finished partitions

    int finished_curr_round;
    int curr_round;
    std::vector<int> finished_rounds;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_curr_round = 0;
        curr_round = 0;
        finished_partitions = 0;
        std::fill(finished_rounds.begin(), finished_rounds.end(), 0);
    }
    // void start();
    // void start_flow(int src_idx, int round);
    void start_flow(int src_idx, int id);
};

void ar_finish_ring(void * arinfo);
void ar_finish_ring_intergpu(void * arinfo);

class FFPSAllreduce;

struct FFPSAllreduceFlow {
    FFPSAllreduce * ar;
    int node_idx;
    int direction;
};

class FFPSAllreduce : public FFTask {

public:
    FFPSAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id,  double local_runtime = 0);
    ~FFPSAllreduce() = default;

    std::vector<uint64_t> node_group; // group of nodes
    uint32_t operator_size;      // total data size of the operator
    int pserver;

    int curr_round;              // will be 2 (scatter, gather)
    std::vector<int> finished_rounds;
    int finished_curr_round;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_curr_round = 0;
        curr_round = 0;
        std::fill(finished_rounds.begin(), finished_rounds.end(), 0);
    }
    void start_flow(int node_idx, int direction);
};

void ar_finish_ps(void * arinfo);


class FFDPSAllreduce;

// struct FFDPSAllreduceFlow {
//     FFDPSAllreduce * ar;
//     int id;
//     int src_idx;
//     int round;
// };

class FFDPSAllreduce : public FFTask {

public:
    FFDPSAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFDPSAllreduce() = default;

    std::vector<uint64_t> node_group; // group of nodes in the order of the ring
    uint32_t operator_size;      // total data size of the operator
    int finished_partitions;     // number of finished partitions

    int finished_curr_round;
    int curr_round;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_curr_round = 0;
        curr_round = 0;
        finished_partitions = 0;
    }
    // void start();
    // void start_flow(int src_idx, int round);
    void start_flow(int src_node, int dst_node);
};

void ar_finish_dps(void * ar_ptr);


class FFReduceScatter;

struct FFReduceScatterFlow {
    FFReduceScatter * rs;
    int id;
    int src_idx;
    int rounds;
};

class FFReduceScatter : public FFTask {
public:
    FFReduceScatter(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFReduceScatter() = default;
    std::vector<uint64_t> node_group; // group of nodes
    uint32_t operator_size;      // total data size of the operator
    int finished_partitions;     // number of finished partitions

    int finished_curr_round;
    int curr_round;
    std::vector<int> finished_rounds;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_curr_round = 0;
        curr_round = 0;
        finished_partitions = 0;
        std::fill(finished_rounds.begin(), finished_rounds.end(), 0);
    }
    void start_flow(int src_idx, int id);
};

void ar_finish_reducescatter(void * arinfo);

class FFAllGather;

struct FFAllGatherFlow {
    FFAllGather * ag;
    int id;
    int src_idx;
    int rounds;
};

class FFAllGather : public FFTask {
public:
    FFAllGather(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFAllGather() = default;
    std::vector<uint64_t> node_group; // group of nodes
    uint32_t operator_size;      // total data size of the operator
    int finished_partitions;     // number of finished partitions

    int finished_curr_round;
    int curr_round;
    std::vector<int> finished_rounds;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_curr_round = 0;
        curr_round = 0;
        finished_partitions = 0;
        std::fill(finished_rounds.begin(), finished_rounds.end(), 0);
    }
    void start_flow(int src_idx, int id);
};

void ar_finish_allgather(void * arinfo);

class FFAlltoAll;

struct FFAlltoAllFlow {
    FFAlltoAll * a2a;
    int src_idx;
    int dst_idx;
    int intra_node_routing = 0;
};

class FFAlltoAll : public FFTask {

public:
    // FFAlltoAll(FFApplication * ffapp, std::vector<uint64_t> ng, uint64_t sz, double local_runtime = 0);
    FFAlltoAll(FFApplication * ffapp, uint64_t taskid, int counter, std::vector<uint64_t> fn, std::vector<uint64_t> tn, std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> o_sizes, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFAlltoAll() = default;

    std::vector<uint64_t> from_node_ids; // group of nodes
    std::vector<uint64_t> to_node_ids; // group of nodes
    std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> operator_sizes;     // data size of the operator
    std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> operator_sizes_ocs;     // data size of the operator

    int finished_partitions;     // number of finished partitions

    int curr_round;
    int total_rounds;
    Matrix2D<double> all2all_traffic_matrix;
    
    // Traffic statistics: per-node volume tracking
    // Sent traffic
    std::unordered_map<int, uint64_t> node_total_volume;  // total volume sent from each node
    std::unordered_map<int, uint64_t> node_ocs_volume;    // OCS volume sent from each node
    std::unordered_map<int, uint64_t> node_eps_volume;    // EPS volume sent from each node
    // Received traffic
    std::unordered_map<int, uint64_t> node_total_recv_volume;  // total volume received at each node
    std::unordered_map<int, uint64_t> node_ocs_recv_volume;    // OCS volume received at each node
    std::unordered_map<int, uint64_t> node_eps_recv_volume;    // EPS volume received at each node
    int flag = 0;
    virtual void doNextEvent();
    virtual void updatetrafficmatrix(); // update traffic matrix for all2all
    virtual void reset() {
        FFTask::reset();
    }
    void start_flow(int src_idx, int dst_idx);
};

void finish_alltoall(void * a2ainfo);

// FFP2P
class FFP2P;

struct FFP2PFlow {
    FFP2P * p2p;
    uint64_t src_idx;
    uint64_t dst_idx;
};

class FFP2P : public FFTask {
public:
    FFP2P(FFApplication * ffapp, uint64_t taskid, int counter, uint64_t src_idx, uint64_t dst_idx, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    FFP2P(FFApplication * ffapp, uint64_t taskid, int counter, std::vector<uint64_t> src_indices, std::vector<uint64_t> dst_indices, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime = 0);
    ~FFP2P() = default;

    std::vector<uint64_t> src_indices;
    std::vector<uint64_t> dst_indices;
    uint64_t operator_size;      // total data size of the operator
    size_t total_flows;
    size_t finished_flows;

    virtual void doNextEvent();
    virtual void reset() {
        FFTask::reset();
        finished_flows = 0;
    }
    void start_flow(uint64_t src_idx, uint64_t dst_idx);
};

void finish_p2p(void * p2pinfo);

class FFApplication {
public:

    enum FFAllReduceStrategy {
        FF_RING_AR,
        FF_PS_AR,
        FF_DPS_AR,
        FF_DEFAULT_AR
    };

    // FFApplication(Topology* top, int cwnd, double pull_rate,  
	// 		NdpRtxTimerScanner & nrts, NdpSinkLoggerSampling & sl, EventList & eventlist, std::string taskgraph);
    // FFApplication(Topology* top, int ss, TcpSinkLoggerSampling & sl, TcpTrafficLogger & tl,
    //     TcpRtxTimerScanner & rtx, EventList & eventlist, std::string taskgraph);
    FFApplication(Topology* top, int ss, string logdir, ofstream * _fstream_out,
        TcpRtxTimerScanner & rtx, EventList & eventlist, FFAllReduceStrategy ars = FFApplication::FF_DEFAULT_AR,
        int is_mixnet_=0, ofstream * _sub_fstream_out=NULL, MixnetTopoManager* _topomanager=NULL);

    FFApplication(Topology* top, int ss, string logdir, ofstream * _fstream_out, std::vector<int> gpus,
        TcpRtxTimerScanner & rtx, EventList & eventlist, FFAllReduceStrategy ars = FFApplication::FF_DEFAULT_AR,
        int is_mixnet_=0, ofstream * _sub_fstream_out=NULL, MixnetTopoManager* _topomanager=NULL);
        
	~FFApplication();

    // void load_taskgraph_json(std::string & taskgraph);
    // void load_taskgraph_protobuf(std::string & taskgraph);
    void load_taskgraph_flatbuf(std::string & taskgraph,std::string & weight_matrix_file);
    void load_taskgraph_flatbuf(std::string & taskgraph);
    void start_init_tasks();

    void reset_and_restart();

    static std::vector<int> choose_gpus(std::unordered_set<int> & candidates, int n);

    static bool LoadFileRaw(const char *name, std::string *buf) {
        std::ifstream ifs(name, std::ifstream::binary);
        if (!ifs.is_open()) {
            return false;
        }
        // The fastest way to read a file into a string.
        ifs.seekg(0, std::ios::end);
        auto size = ifs.tellg();
        (*buf).resize(static_cast<size_t>(size));
        ifs.seekg(0, std::ios::beg);
        ifs.read(&(*buf)[0], (*buf).size());
        if (ifs.bad()) return false;
        ifs.close();
        return true;
    }

    size_t nnodes, ngpupernode, nswitches;
    
	int cwnd;
	double pull_rate;
    std::unordered_map<uint64_t, FFTask*> tasks;
    std::unordered_map<uint64_t, FFDevice*> devices;
	Topology * topology; 
    int ssthresh;
    EventList & eventlist;
    unordered_map<uint64_t, unsigned int> counters;
	// NdpRtxTimerScanner & ndpRtxScanner;
	// NdpSinkLoggerSampling & sinkLogger;
    // TcpSinkLoggerSampling & sinkLogger;
    // TcpTrafficLogger & tcpTrafficLogger;
    FFAllReduceStrategy allreduce_strategy;
    string logdir;
    ofstream * fstream_out;
    ofstream * sub_fstream_out;
    TcpRtxTimerScanner & tcpRtxScanner;
    std::unordered_map<uint64_t, std::vector<std::vector<int>>> selected_jumps; // useless here
    bool fancy_ring;
    bool finished_once;

    static int total_apps;
    static int finished_apps;

    std::vector<int> gpus;
    simtime_picosec final_finish_time;
    simtime_picosec first_iter_time;
    size_t n_finished_tasks;

    double nvlink_bandwidth;//byte

    int dp_degree;
    int tp_degree;
    int pp_degree;
    int ep_degree;
    int is_mixnet;
    std::vector<std::vector<int>> weight_matrix;
    MixnetTopoManager* topomanager; // we use this to manage the mixnet topology

    std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> global_operator_sizes;

    //moe reconfig
    // traffic matrix
    // 
};


void taskfinish(void * task);

#endif // FF_APP_H
