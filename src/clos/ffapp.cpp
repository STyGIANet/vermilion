#include <fstream>
#include <streambuf>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <assert.h>
#include <queue>
#include <set>
#include <utility>

#include "ffapp.h"
#include "ndp.h"
#include "dctcp.h"
#include "route.h"
#include "json.hpp"

#define DEFAULT_PACKET_SIZE 1500 // full packet (including header), Bytes
#define NVLINK_BANDWIDTH (600.0 * 1024 * 1024 * 1024)

using json = nlohmann::json;

namespace {
void dump_region_matrix(std::ostream& os,
                        const std::vector<std::vector<uint64_t>>& matrix) {
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                os << ' ';
            }
            os << row[i];
        }
        os << std::endl;
    }
}
}

int FFApplication::total_apps = 0;
int FFApplication::finished_apps = 0;
int FFApplication::debug_exit;
bool check_maps_equal(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map1,
                      const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map2) {
    if (map1.size() != map2.size()) {
        return false;
    }
    for (const auto& entry : map1) {
        auto it = map2.find(entry.first);
        if (it == map2.end()) {
            return false;
        }
        if (it->second != entry.second) {
            return false;
        }
    }
    return true;
}

bool check_symmetric_positions_equal(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map1,
                                     const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> map2) {
    for (const auto& entry : map1) {
        int i = entry.first.first;
        int j = entry.first.second;

        auto it = map2.find(std::make_pair(j, i));
        if (it == map2.end()) {
            return false;
        }
        if (entry.second != it->second) {
            return false;
        }
    }
    return true;
}

void print_operator_sizes_matrix(const std::unordered_map<std::pair<int, int>, uint64_t, pair_hash>& operator_sizes, int matrix_size, int tp_degree) {
    (void)matrix_size;
    if (operator_sizes.empty()) {
        std::cout << "(empty operator_sizes)" << std::endl;
        return;
    }

    std::set<int> row_groups;
    std::set<int> col_groups;
    for (const auto& entry : operator_sizes) {
        row_groups.insert(entry.first.first / tp_degree);
        col_groups.insert(entry.first.second / tp_degree);
    }

    std::vector<int> row_indices(row_groups.begin(), row_groups.end());
    std::vector<int> col_indices(col_groups.begin(), col_groups.end());

    std::cout << "\t";
    for (int col_group : col_indices) {
        std::cout << col_group * tp_degree << "\t";
    }
    std::cout << std::endl;

    for (int row_group : row_indices) {
        std::cout << row_group * tp_degree << "\t";
        for (int col_group : col_indices) {
            uint64_t aggregated = 0;
            for (int tp_idx = 0; tp_idx < tp_degree; ++tp_idx) {
                auto key = std::make_pair(row_group * tp_degree + tp_idx,
                                          col_group * tp_degree + tp_idx);
                auto it = operator_sizes.find(key);
                if (it != operator_sizes.end()) {
                    aggregated += it->second;
                }
            }
            std::cout << aggregated << "\t";
        }
        std::cout << std::endl;
    }
}

FFApplication::FFApplication(Topology* top, int ss, string logdir, ofstream * _fstream_out, //TcpSinkLoggerSampling & sl, TcpTrafficLogger & tl,
    TcpRtxTimerScanner & rtx, EventList & eventlist, FFAllReduceStrategy ars, int is_mixnet_, ofstream * _sub_fstream_out, MixnetTopoManager* _topomanager)
: topology(top), ssthresh(ss), eventlist(eventlist), logdir(logdir),
  fstream_out(_fstream_out), tcpRtxScanner(rtx), 
  final_finish_time(0), n_finished_tasks(0), allreduce_strategy(ars), sub_fstream_out(_sub_fstream_out), topomanager(_topomanager) {
    // std::cerr << "allreduce_strategy: " << allreduce_strategy << std::endl;
    FFApplication::total_apps++;
    fancy_ring = false;
    finished_once = false;
    nvlink_bandwidth = NVLINK_BANDWIDTH; // 600GB, byte
    is_mixnet=is_mixnet_;
}

FFApplication::FFApplication(Topology* top, int ss, string logdir, ofstream * _fstream_out, std::vector<int> gpus,
    TcpRtxTimerScanner & rtx, EventList & eventlist, FFAllReduceStrategy ars,int is_mixnet_,ofstream * _sub_fstream_out, MixnetTopoManager* _topomanager)
: topology(top), ssthresh(ss), eventlist(eventlist), logdir(logdir), 
  fstream_out(_fstream_out), gpus(gpus), tcpRtxScanner(rtx), 
  final_finish_time(0), n_finished_tasks(0), allreduce_strategy(ars),sub_fstream_out(_sub_fstream_out), topomanager(_topomanager) {
    // std::cerr << "allreduce_strategy: " << allreduce_strategy << std::endl;
    FFApplication::total_apps++;
    fancy_ring = false;
    finished_once = false;
    nvlink_bandwidth = NVLINK_BANDWIDTH; // 600GB, byte
    is_mixnet=is_mixnet_;
}

FFApplication::~FFApplication() {

    for (auto item: tasks) {
        delete item.second;
    }
    for (auto item: devices) {
        delete item.second;
    }

}
class TaskCompare {
public:
  bool operator()(FFTask *lhs, FFTask *rhs) {
    return lhs->ready_time > rhs->ready_time;
  }
};
std::vector<int> FFApplication::choose_gpus(std::unordered_set<int> & candidates, int n)
{
    std::vector<int> result;
    assert(n <= candidates.size());
    std::sample(candidates.begin(), candidates.end(), std::back_inserter(result),
        n, std::mt19937{std::random_device{}()});
    for (int i : result) {
        candidates.erase(candidates.find(i));
    }
    return result;
}
void load_weight_matrix(std::string & weight_matrix_file, std::vector<std::vector<int>> &weight_matrix){
    std::ifstream file(weight_matrix_file);
    std::string line;
    if (file.is_open()) {
        std::getline(file, line);
        file.close();
        line = line.substr(1, line.size() - 2);
        std::stringstream ss(line);
        std::string block;

        while (std::getline(ss, block, ']')) {
            block = block.substr(block.find('[') + 1);
            std::stringstream blockStream(block);
            std::string number;
            std::vector<int> row;
            int total_row_number=0;
            while (std::getline(blockStream, number, ',')) {
                row.push_back(std::stoi(number));
                total_row_number+=std::stoi(number);
            }
            assert(total_row_number==32768);
            weight_matrix.push_back(row);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
}

void FFApplication::load_taskgraph_flatbuf(std::string & taskgraph, std::string & weight_matrix_file) {
    string buffer;
    bool success = FFApplication::LoadFileRaw(taskgraph.c_str(), &buffer);
    if (!success) {
        assert("Failed to read file!" && false);
    }
    load_weight_matrix(weight_matrix_file, weight_matrix);
    flatbuffers::Verifier::Options opts;
    opts.max_tables = 10000000;  // Increase to 10 million
    opts.max_depth = 64;
    // Verify the FlatBuffer before using it
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(), opts);
    bool verification_result = verifier.VerifyBuffer<FlatBufTaskGraph::TaskGraph>(nullptr);
    if (!verification_result) {
        std::cerr << "ERROR: FlatBuffer verification failed for file: " << taskgraph << std::endl;
        std::cerr << "File size: " << buffer.size() << " bytes" << std::endl;
        std::cerr << "This usually means the file is corrupted, incomplete, or uses an incompatible schema version." << std::endl;
        assert("FlatBuffer verification failed!" && false);
    }
    
    auto fbuf_tg = flatbuffers::GetRoot<FlatBufTaskGraph::TaskGraph>(buffer.c_str());
    ngpupernode = fbuf_tg->ngpupernode();
    nnodes = fbuf_tg->nnode();
    dp_degree=fbuf_tg->dp_degree();
    tp_degree=fbuf_tg->tp_degree();
    pp_degree=fbuf_tg->pp_degree();
    ep_degree=fbuf_tg->ep_degree();
    std::cerr << " dp_degree " << dp_degree << " tp_degree " << tp_degree << " pp_degree " << pp_degree << " ep_degree " << ep_degree<<std::endl;
    if (gpus.empty()) {
        gpus.resize(nnodes);
        std::iota(std::begin(gpus), std::end(gpus), 0);
    }

    // load device 
    for (int i = 0; i < fbuf_tg->devices()->size(); i++) {
        auto dev = fbuf_tg->devices()->Get(i);
        std::cerr << " device id " << dev->nodeid() <<std::endl;
        devices[dev->deviceid()] = new FFDevice(
            this,
            dev->type(),
            dev->nodeid(),
            dev->deviceproperty(),
            dev->bandwidth()
        );
    }

    // load tasks
    std::cerr << "load_taskgraph_flatbuf: start load tasks"<<endl;
    if (!fbuf_tg->tasks()) {
        std::cerr << "ERROR: tasks() is null!" << std::endl;
        assert("tasks() is null!" && false);
    }
    std::cerr << "Total tasks to load: " << fbuf_tg->tasks()->size() << endl;
    
    for (int i = 0; i < fbuf_tg->tasks()->size(); i++) {
        auto task_ptr = fbuf_tg->tasks()->Get(i);
        if (!task_ptr) {
            std::cerr << "ERROR: task pointer at index " << i << " is null!" << std::endl;
            assert("task pointer is null!" && false);
        }
        auto &this_task = *task_ptr;
        //std::cerr << "Loading task index " << i << ", task id "<<this_task.taskid()<<endl;
        assert(tasks.find(this_task.taskid()) == tasks.end());
        if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_SUB_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_DP_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_TP_ALLREDUCE) {
            std::cout << "LOGTASK AllReduce" << std::endl;
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                //node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
                auto fn = this_task.to_node_ids()->Get(j);
                node_group.push_back(fn);
            }
            
            if (fancy_ring) {
                // cout << "fancy" << endl;
                tasks[this_task.taskid()] = new FFNewRingAllreduce(
                    this,
                    this_task.taskid(), 
                    node_group, 
                    selected_jumps[node_group.size()],
                    this_task.counter(),
                    this_task.xfersize(),
                    this_task.info()->str(),
                    this_task.micro_batch_id(),
                    this_task.layer_id(),
                    this_task.target_micro_batch_id(),
                    this_task.target_layer_id(),
                    this_task.runtime()
                );
            }
            else {
                if (allreduce_strategy == FFApplication::FF_RING_AR || 
                    allreduce_strategy == FFApplication::FF_DEFAULT_AR) 
                {
                    // cout << "ring" << endl;
                    tasks[this_task.taskid()] = new FFRingAllreduce(
                        this,
                        this_task.taskid(),
                        node_group, 
                        this_task.counter(),
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                }
                else if (allreduce_strategy == FFApplication::FF_PS_AR)
                {
                    // cout << "ps" << endl;
                    tasks[this_task.taskid()] = new FFPSAllreduce(
                        this,
                        this_task.taskid(),
                        node_group,
                        this_task.counter(), 
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                }
                else if (allreduce_strategy == FFApplication::FF_DPS_AR) {
                    // cout << "dps" << endl;
                    tasks[this_task.taskid()] = new FFDPSAllreduce(
                        this,
                        this_task.taskid(),
                        node_group, 
                        this_task.counter(),
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                } 
            }
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_REDUCESCATTER) {
            std::cout << "LOGTASK ReduceScatter" << std::endl;
            // Ring reducescatter
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
            }
            tasks[this_task.taskid()] = new FFReduceScatter(
                this, 
                this_task.taskid(),
                node_group, 
                this_task.counter(),
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLGATHER) {
            std::cout << "LOGTASK AllGather" << std::endl;
            // Ring allgather
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
            }
            tasks[this_task.taskid()] = new FFAllGather(
                this,
                this_task.taskid(),
                node_group, 
                this_task.counter(),
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLTOALL) {
            std::cout << "LOGTASK AlltoAll" << std::endl;

            std::vector<uint64_t> from_node;
            std::vector<uint64_t> to_node;
            std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> operator_sizes;
            for (int j = 0; j < this_task.from_node_ids()->size(); j++) {
                auto fn = this_task.from_node_ids()->Get(j);
                from_node.push_back(fn);
            }
            for (int k = 0; k < this_task.to_node_ids()->size(); k++) {
                auto tn = this_task.to_node_ids()->Get(k);
                to_node.push_back(tn);
            }

            uint64_t total_xfer_size=this_task.xfersize()*ep_degree;//no need to x tp_degree
            //std::cout<<"New_log Total xfer size for task id "<<this_task.taskid()<<" is "<<total_xfer_size<< "task layer id " << this_task.layer_id() << std::endl;
            int is_first=0;
            if(global_operator_sizes.empty()) {
                is_first=1;
            }
            for (int j = 0; j < this_task.from_node_ids()->size()/tp_degree; j++) {
                for (int k = 0; k < this_task.to_node_ids()->size()/tp_degree; k++) {
                    for(int tp_idx=0; tp_idx<tp_degree; tp_idx++){
                        auto fn = this_task.from_node_ids()->Get(j*tp_degree+tp_idx);
                        auto tn = this_task.to_node_ids()->Get(k*tp_degree+tp_idx);
                        auto fn_expid=(fn/tp_degree)%ep_degree; //ep*tp
                        auto tn_expid=(tn/tp_degree)%ep_degree;
                        if((this_task.info()->str().find("GROUP_BY")!= std::string::npos && this_task.info()->str().find("forward")!= std::string::npos)
                        || (this_task.info()->str().find("AGGREGATE")!= std::string::npos && this_task.info()->str().find("backward")!= std::string::npos)) {
                            float xfer_size_tmp=0;
                            xfer_size_tmp = total_xfer_size * 1.0 * weight_matrix[fn_expid%8][tn_expid%8]/32768;
                            operator_sizes[std::make_pair(fn, tn)] =  static_cast<uint64_t>(std::ceil(xfer_size_tmp));
                        }
                        else {
                            float xfer_size_tmp=0;
                            xfer_size_tmp = total_xfer_size*1.0*weight_matrix[tn_expid%8][fn_expid%8]/32768;
                            operator_sizes[std::make_pair(fn, tn)] =  static_cast<uint64_t>(std::ceil(xfer_size_tmp));
                        }
                        if(is_first)
                        {
                            global_operator_sizes[std::make_pair(fn, tn)] = operator_sizes[std::make_pair(fn, tn)];
                        }
                        //Deeksha: Debugging 
                        //cout<<" from_node "<<fn<<" to_node "<<tn<<" size "<<operator_sizes[std::make_pair(fn, tn)]<<"tp degree" <<tp_degree<<"EP Degree"<<ep_degree<<endl;
                        cout<<" log3 task id "<<this_task.taskid()<<" from_node "<<fn<<" to_node "<<tn<<" size "<<operator_sizes[std::make_pair(fn, tn)]<<" tp degree" <<tp_degree<<" EP Degree"<<ep_degree<<endl;
                    }
                }

            }
            //cout<<" new_log Finished populating operator sizes for task id "<<this_task.taskid()<<" task size "<< " Total from" << this_task.from_node_ids()->size()<<" to " << this_task.to_node_ids()->size()<<" total size "<<total_xfer_size<<endl;
            std::cerr << " operator_sizes summary: total_xfer_size=" << total_xfer_size
                      << " from_node_groups=" << this_task.from_node_ids()->size() / tp_degree
                      << " micro_batch_id=" << this_task.micro_batch_id()
                      << " info=" << this_task.info()->str()
                      << std::endl;
            std::cerr << " operator_sizes matrix" <<endl;
            print_operator_sizes_matrix(operator_sizes,this_task.from_node_ids()->size(),tp_degree);

            tasks[this_task.taskid()] = new FFAlltoAll(
                this,
                this_task.taskid(), 
                this_task.counter(),
                from_node,
                to_node, 
                operator_sizes,
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
            cout<<" log1 task id "<<this_task.taskid()<<" created alltoall task "<<tasks[this_task.taskid()]->name<< " microbatch id " << this_task.micro_batch_id()<<" layer id " << this_task.layer_id() << " transfer size " << total_xfer_size<<endl;
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_P2P) {
            std::cout << "LOGTASK P2P" << std::endl;
            // P2P
            std::vector<uint64_t> from_nodes;
            std::vector<uint64_t> to_nodes;
            for (int j = 0; j < this_task.from_node_ids()->size(); j++) {
                auto fn = this_task.from_node_ids()->Get(j);
                from_nodes.push_back(fn);
            }
            for (int j = 0; j < this_task.to_node_ids()->size(); j++) {
                auto tn = this_task.to_node_ids()->Get(j);
                to_nodes.push_back(tn);
            }
            assert(from_nodes.size() == to_nodes.size());
            tasks[this_task.taskid()] = new FFP2P(
                this, 
                this_task.taskid(),
                this_task.counter(),
                from_nodes,
                to_nodes, 
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else {
            std::cout << "LOGTASK Default" << std::endl;
            tasks[this_task.taskid()] = new FFTask(
                this, 
                this_task.type(), 
                devices[this_task.deviceid()],
                this_task.taskid(), 
                this_task.xfersize(), 
                this_task.runtime(),
                this_task.counter(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id()
            );
                
            
        }

        for (int j = 0; j < this_task.nexttasks()->size(); j++) {
            uint64_t next_id = this_task.nexttasks()->Get(j);
            tasks[this_task.taskid()]->next_tasks.push_back(next_id);
        }
        counters[this_task.taskid()] = this_task.counter();
        tasks[this_task.taskid()]->name=this_task.name()->str();
    }
}

void FFApplication::load_taskgraph_flatbuf(std::string & taskgraph) {
    // we dont use it anymore
    string buffer;
    bool success = FFApplication::LoadFileRaw(taskgraph.c_str(), &buffer);
    if (!success) {
        assert("Failed to read file!" && false);
    }
    
    // Verify the FlatBuffer before using it
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
    bool verification_result = verifier.VerifyBuffer<FlatBufTaskGraph::TaskGraph>(nullptr);
    if (!verification_result) {
        std::cerr << "ERROR: FlatBuffer verification failed for file: " << taskgraph << std::endl;
        std::cerr << "File size: " << buffer.size() << " bytes" << std::endl;
        std::cerr << "This usually means the file is corrupted, incomplete, or uses an incompatible schema version." << std::endl;
        assert("FlatBuffer verification failed!" && false);
    }
    
    auto fbuf_tg = flatbuffers::GetRoot<FlatBufTaskGraph::TaskGraph>(buffer.c_str());
    ngpupernode = fbuf_tg->ngpupernode();
    nnodes = fbuf_tg->nnode();
    dp_degree=fbuf_tg->dp_degree();
    tp_degree=fbuf_tg->tp_degree();
    pp_degree=fbuf_tg->pp_degree();
    ep_degree=fbuf_tg->ep_degree();
    std::cerr << " dp_degree " << dp_degree << " tp_degree " << tp_degree << " pp_degree " << pp_degree << " ep_degree " << ep_degree<<std::endl;
    if (gpus.empty()) {
        gpus.resize(nnodes);
        std::iota(std::begin(gpus), std::end(gpus), 0);
    }

    // load device 
    for (int i = 0; i < fbuf_tg->devices()->size(); i++) {
        auto dev = fbuf_tg->devices()->Get(i);
        devices[dev->deviceid()] = new FFDevice(
            this,
            dev->type(),
            dev->nodeid(),
            dev->deviceproperty(),
            dev->bandwidth()
        );
    }

    // load tasks
    std::cerr << "load_taskgraph_flatbuf: start load tasks"<<endl;
    for (int i = 0; i < fbuf_tg->tasks()->size(); i++) {
        auto &this_task = *fbuf_tg->tasks()->Get(i);
        std::cerr << "task id "<<this_task.taskid()<<endl;
        assert(tasks.find(this_task.taskid()) == tasks.end());
        if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_SUB_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_DP_ALLREDUCE ||
            this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_TP_ALLREDUCE) {
            
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                //node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
                auto fn = this_task.to_node_ids()->Get(j);
                node_group.push_back(fn);
            }
            
            if (fancy_ring) {
                // cout << "fancy" << endl;
                tasks[this_task.taskid()] = new FFNewRingAllreduce(
                    this,
                    this_task.taskid(), 
                    node_group, 
                    selected_jumps[node_group.size()],
                    this_task.counter(),
                    this_task.xfersize(),
                    this_task.info()->str(),
                    this_task.micro_batch_id(),
                    this_task.layer_id(),
                    this_task.target_micro_batch_id(),
                    this_task.target_layer_id(),
                    this_task.runtime()
                );
            }
            else {
                if (allreduce_strategy == FFApplication::FF_RING_AR || 
                    allreduce_strategy == FFApplication::FF_DEFAULT_AR) 
                {
                    // cout << "ring" << endl;
                    tasks[this_task.taskid()] = new FFRingAllreduce(
                        this,
                        this_task.taskid(),
                        node_group, 
                        this_task.counter(),
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                }
                else if (allreduce_strategy == FFApplication::FF_PS_AR)
                {
                    // cout << "ps" << endl;
                    tasks[this_task.taskid()] = new FFPSAllreduce(
                        this,
                        this_task.taskid(),
                        node_group,
                        this_task.counter(), 
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                }
                else if (allreduce_strategy == FFApplication::FF_DPS_AR) {
                    // cout << "dps" << endl;
                    tasks[this_task.taskid()] = new FFDPSAllreduce(
                        this,
                        this_task.taskid(),
                        node_group, 
                        this_task.counter(),
                        this_task.xfersize(),
                        this_task.info()->str(),
                        this_task.micro_batch_id(),
                        this_task.layer_id(),
                        this_task.target_micro_batch_id(),
                        this_task.target_layer_id(),
                        this_task.runtime()
                    );
                } 
            }
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_REDUCESCATTER) {
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
            }
            tasks[this_task.taskid()] = new FFReduceScatter(
                this, 
                this_task.taskid(),
                node_group, 
                this_task.counter(),
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLGATHER) {
            std::vector<uint64_t> node_group;
            for (int j = 0; j < this_task.to_node_ids()->size() ; j++) {
                node_group.push_back(reinterpret_cast<uint64_t>(this_task.to_node_ids()->Get(j)));
            }
            tasks[this_task.taskid()] = new FFAllGather(
                this,
                this_task.taskid(),
                node_group, 
                this_task.counter(),
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_ALLTOALL) {

            std::vector<uint64_t> from_node;
            std::vector<uint64_t> to_node;
            std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> operator_sizes;
            for (int j = 0; j < this_task.from_node_ids()->size(); j++) {
                auto fn = this_task.from_node_ids()->Get(j);
                from_node.push_back(fn);
            }
            for (int k = 0; k < this_task.to_node_ids()->size(); k++) {
                auto tn = this_task.to_node_ids()->Get(k);
                to_node.push_back(tn);
            }
            /*
                p2p pattern
                expert2expert: p2p
                tp group inside expert: one2one
            */
            int is_groupby=0;
            if(this_task.info()->str().find("GROUP_BY")!= std::string::npos){
                is_groupby=1;
            }
            else{
                assert(this_task.info()->str().find("AGGREGATE")!= std::string::npos);
            }
            int total_xfer_size=this_task.xfersize()*ep_degree;//no need to x tp_degree

            for (int j = 0; j < this_task.from_node_ids()->size()/tp_degree; j++) {
                for (int k = 0; k < this_task.to_node_ids()->size()/tp_degree; k++) {
                    for(int tp_idx=0; tp_idx<tp_degree; tp_idx++){
                        auto fn = this_task.from_node_ids()->Get(j*tp_degree+tp_idx);
                        auto tn = this_task.to_node_ids()->Get(k*tp_degree+tp_idx);
                        operator_sizes[std::make_pair(fn, tn)] = this_task.xfersize();
                    }
                }
            }
            tasks[this_task.taskid()] = new FFAlltoAll(
                this,
                this_task.taskid(), 
                this_task.counter(),
                from_node,
                to_node, 
                operator_sizes,
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else if (this_task.type() == FlatBufTaskGraph::SimTaskType_TASK_P2P) {
            // P2P
            std::vector<uint64_t> from_nodes;
            std::vector<uint64_t> to_nodes;
            for (int j = 0; j < this_task.from_node_ids()->size(); j++) {
                auto fn = this_task.from_node_ids()->Get(j);
                from_nodes.push_back(fn);
            }
            for (int j = 0; j < this_task.to_node_ids()->size(); j++) {
                auto tn = this_task.to_node_ids()->Get(j);
                to_nodes.push_back(tn);
            }
            assert(from_nodes.size() == to_nodes.size());
            tasks[this_task.taskid()] = new FFP2P(
                this, 
                this_task.taskid(),
                this_task.counter(),
                from_nodes,
                to_nodes, 
                this_task.xfersize(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id(),
                this_task.runtime()
            );
        }
        else {
            tasks[this_task.taskid()] = new FFTask(
                this, 
                this_task.type(), 
                devices[this_task.deviceid()],
                this_task.taskid(), 
                this_task.xfersize(), 
                this_task.runtime(),
                this_task.counter(),
                this_task.info()->str(),
                this_task.micro_batch_id(),
                this_task.layer_id(),
                this_task.target_micro_batch_id(),
                this_task.target_layer_id()
            );
                
            
        }

        for (int j = 0; j < this_task.nexttasks()->size(); j++) {
            uint64_t next_id = this_task.nexttasks()->Get(j);
            tasks[this_task.taskid()]->next_tasks.push_back(next_id);
        }
        counters[this_task.taskid()] = this_task.counter();
        tasks[this_task.taskid()]->name=this_task.name()->str();
    }  
}


void FFApplication::start_init_tasks() {
    simtime_picosec delta = 0;
    int count = 0;
    for (auto task: tasks) {
        FFTask * t = task.second;
        if (t->counter == 0) {
            t->state = FFTask::TASK_READY;
            t->eventlist().sourceIsPending(*t, t->eventlist().now() + delta++);
            count++;
        }
    }
    std::cerr << "added " << count << " init tasks." << std::endl;
}

void FFApplication::reset_and_restart() {
    n_finished_tasks = 0;
    for (auto task: tasks) {
        task.second->reset();
    }
    for (auto item: counters) {
        tasks[item.first]->counter = item.second;
    }
    start_init_tasks();
}

FFTask::FFTask(FFApplication * ffapp, std::string type, FFDevice * device, uint64_t taskid, uint64_t xfersize, 
    float runtime, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id): ffapp(ffapp), EventSource(ffapp->eventlist, "FFTask") {

    if (type == "TASK_FORWARD") {
        this->type = FFTaskType::TASK_FORWARD;
    }
    else if (type == "TASK_BACKWARD") {
        this->type = FFTaskType::TASK_BACKWARD;
    }
    else if (type == "TASK_COMM") {
        this->type = FFTaskType::TASK_COMM;
    }
    else if (type == "TASK_UPDATE") {
        this->type = FFTaskType::TASK_UPDATE;
    }
    else if (type == "TASK_BARRIER") {
        this->type = FFTaskType::TASK_BARRIER;
    }
    else if (type == "TASK_P2P") {
        this->type = FFTaskType::TASK_P2P;
    }
    else if (type == "TASK_NOMINAL_COMM") {
        this->type = FFTaskType::TASK_NOMINAL_COMM;
    }
    else {
        throw "Unsupported task type!";
    }

    this->state = TASK_NOT_READY;
    this->device = device;
    this->run_time = runtime * 1000000000ULL;
    this->taskid = taskid;
    this->xfersize = xfersize;
    this->counter = counter;

    this->info = info;
    this->micro_batch_id = micro_batch_id;
    this->layer_id = layer_id;
    this->target_micro_batch_id = target_micro_batch_id;
    this->target_layer_id = target_layer_id;

    if (device->type == FFDevice::DEVICE_NW_COMM) {
        this->src_node = device->from_node;
        this->dst_node = device->to_node;
    }
    else {
        this->src_node = this->dst_node = -1; 
    }

    ready_time = 0;
    start_time = 0;
    finish_time = 0;
    // counter = 0;
}

FFTask::FFTask(FFApplication * ffapp, FlatBufTaskGraph::SimTaskType tasktype, FFDevice * device, uint64_t taskid,
     uint64_t xfersize, float runtime, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id): ffapp(ffapp), EventSource(ffapp->eventlist, "FFTask") {
    
    if (tasktype == FlatBufTaskGraph::SimTaskType_TASK_FORWARD) {
        this->type = FFTaskType::TASK_FORWARD;
    }
    else if (tasktype == FlatBufTaskGraph::SimTaskType_TASK_BACKWARD) {
        this->type = FFTaskType::TASK_BACKWARD;
    }
    else if (tasktype == FlatBufTaskGraph::SimTaskType_TASK_NOMINAL_COMM) {
        this->type = FFTaskType::TASK_COMM;
        // std::cerr << "adding COMM " << std::endl;
    }
    else if (tasktype == FlatBufTaskGraph::SimTaskType_TASK_UPDATE) {
        this->type = FFTaskType::TASK_UPDATE;
    }
    else if (tasktype == FlatBufTaskGraph::SimTaskType_TASK_BARRIER) {
        this->type = FFTaskType::TASK_BARRIER;
    }
    else {
        throw "Unsupported task type!";
    }

    this->state = TASK_NOT_READY;
    this->device = device;
    this->run_time = runtime * 1000000000ULL;
    this->taskid = taskid;
    this->xfersize = xfersize;
    this->counter = counter;

    this->info = info;
    this->micro_batch_id = micro_batch_id;
    this->layer_id = layer_id;
    this->target_micro_batch_id = target_micro_batch_id;
    this->target_layer_id = target_layer_id;

    if (device->type == FFDevice::DEVICE_NW_COMM) {
        this->src_node = device->from_node;
        this->dst_node = device->to_node;
    }
    else {
        this->src_node = this->dst_node = -1; 
    }

    ready_time = 0;
    start_time = 0;
    finish_time = 0;
    // counter = 0;
}

#if 0
FFTask::FFTask(TaskGraphProtoBuf::Task_SimTaskType type, FFDevice * device, 
         uint64_t xfersize, float runtime): EventSource(ffapp->eventlist, "FFTask") {

    if (type == TaskGraphProtoBuf::Task_SimTaskType_TASK_FORWARD) {
        this->type = FFTaskType::TASK_FORWARD;
    }
    else if (type == TaskGraphProtoBuf::Task_SimTaskType_TASK_BACKWARD) {
        this->type = FFTaskType::TASK_BACKWARD;
    }
    else if (type == TaskGraphProtoBuf::Task_SimTaskType_TASK_COMM) {
        this->type = FFTaskType::TASK_COMM;
    }
    else if (type == TaskGraphProtoBuf::Task_SimTaskType_TASK_UPDATE) {
        this->type = FFTaskType::TASK_UPDATE;
    }
    else if (type == TaskGraphProtoBuf::Task_SimTaskType_TASK_BARRIER) {
        this->type = FFTaskType::TASK_BARRIER;
    }
    else {
        throw "Unsupported task type!";
    }

    this->state = TASK_NOT_READY;
    this->device = device;
    this->run_time = runtime * 1000000000ULL;
    this->xfersize = xfersize;

    if (device->type == FFDevice::DEVICE_NW_COMM) {
        this->src_node = device->from_node;
        this->dst_node = device->to_node;
    }
    else {
        this->src_node = this->dst_node = -1; 
    }

    ready_time = 0;
    start_time = 0;
    finish_time = 0;
    counter = 0;
}
#endif

FFTask::FFTask(FFApplication * ffapp, FFTask::FFTaskType type, uint64_t taskid, int counter, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id):
    ffapp(ffapp), EventSource(ffapp->eventlist, "FFTask") {

    this->type = type;
    this->taskid = taskid;
    this->counter = counter;

    this->info = info;
    this->micro_batch_id = micro_batch_id;
    this->layer_id = layer_id;
    this->target_micro_batch_id = target_micro_batch_id;
    this->target_layer_id = target_layer_id;
}

void FFTask::taskstart() {
    // std::cerr << "Guid: " << guid << " try start at " << eventlist().now() << std::endl;
    assert(counter == 0);
    //finished_task_logger
    if (type == FFTask::TASK_COMM && device->type == FFDevice::DEVICE_NW_COMM) {
        start_flow();
    } 
    else {
        execute_compute();
    }
}

void FFTask::execute_compute() {
    // for mixnet we need to trigger reconfiguration when doing computation to achieve overlapping
    if (this->state == FFTask::TASK_NOT_READY) {
        std::cerr << "ERROR: Executing not ready task!" << std::endl;
        assert(false);
    }

    if (this->state == FFTask::TASK_FINISHED) {
        std::cerr << "ERROR: Executing finished task!" << std::endl;
        assert(false);
    }

    // check if the device has running task. If not, schedule this task.
    // Otherwise schedule it at the time the other task finishes. 
    if (this->state == FFTask::TASK_READY) {
        if (device->state == FFDevice::DEVICE_IDLE) {
            // std::cerr << "Task " << (uint64_t)this << " starts at " << eventlist().now() << std::endl;
            this->state = FFTask::TASK_RUNNING;
            device->state = FFDevice::DEVICE_BUSY;
            start_time = eventlist().now();
            finish_time = start_time + run_time;
            eventlist().sourceIsPending(*this, finish_time);
            device->busy_up_to = finish_time;
            /*
                1. check if this topology is mixnet && check the topology manager is not nullptr
                2. if this task is backward aggregation task, we can trigger regional reconfiguration
                3. only the first regional reconfiguration is allowed
            */
            if (ffapp->is_mixnet && ffapp->topomanager) {
                if (this->name.find("Aggregate") != std::string::npos && this->type == FFTask::TASK_BACKWARD) {
                    int region_id = (this->device->gpu_id / 8) / ffapp->topomanager->region_size;
                    int layer_id = this->layer_id;
                    int micro_batch_id = this->micro_batch_id;
                    if (ffapp->topomanager->regional_topo_managers[region_id]->current_layer_id != layer_id || ffapp->topomanager->regional_topo_managers[region_id]->current_micro_batch_id != micro_batch_id) {
                        // trigger reconfiguration
                        ffapp->topomanager->regional_topo_managers[region_id]->current_layer_id = layer_id;
                        ffapp->topomanager->regional_topo_managers[region_id]->current_micro_batch_id = micro_batch_id;
                        // reset so periodic reconf starts fresh for the new layer
                        ffapp->topomanager->regional_topo_managers[region_id]->first_reconf_done = false;
                        assert(ffapp->topomanager->demandrecorder->check_traffic_matrix(layer_id, region_id));// it must be true
                        eventlist().sourceIsPending(*(ffapp->topomanager->regional_topo_managers[region_id]), eventlist().now());
                       // cout << "backward aggregated task trigger reconfig at "<< eventlist().now() << std::endl;
                    }
                    else {
                        // do nothing, the reconfig is already done
                    }
                }
            }
        }
        else {
            eventlist().sourceIsPending(*this, device->busy_up_to);
        }
    }
    // This means this task has finished
    else if (this->state == FFTask::TASK_RUNNING) {
        // std::cerr << "Task " << (uint64_t)this << " finishes at " << eventlist().now() << std::endl;
        assert(device->state == FFDevice::DEVICE_BUSY);

        this->state = FFTask::TASK_FINISHED;
        device->state = FFDevice::DEVICE_IDLE;

        cleanup();
    }
}
string FFTask::get_string_type(){
    switch (type) {
        case TASK_FORWARD: return "TASK_FORWARD";
        case TASK_BACKWARD: return "TASK_BACKWARD";
        case TASK_COMM: return "TASK_COMM";
        case TASK_UPDATE: return "TASK_UPDATE";
        case TASK_BARRIER: return "TASK_BARRIER";
        case TASK_NOMINAL_COMM: return "TASK_NOMINAL_COMM";
        case TASK_P2P: return "TASK_P2P";
        case TASK_SUB_ALLREDUCE: return "TASK_SUB_ALLREDUCE";
        case TASK_DP_ALLREDUCE: return "TASK_DP_ALLREDUCE";
        case TASK_TP_ALLREDUCE: return "TASK_TP_ALLREDUCE";
        case TASK_ALLREDUCE: return "TASK_ALLREDUCE";
        case TASK_REDUCESCATTER: return "TASK_REDUCESCATTER";
        case TASK_ALLGATHER: return "TASK_ALLGATHER";
        case TASK_ALLTOALL: return "TASK_ALLTOALL";
        default: return "UNKNOWN_TASK";
    }
}
void FFTask::cleanup() {
    this->state = FFTask::TASK_FINISHED;
    ffapp->n_finished_tasks++;
    std::cerr << ffapp << " finished task:" << this->taskid << ", nfin " << ffapp->n_finished_tasks << " ntot " << ffapp->tasks.size() <<" name: "<<this->name<< " type " << this->type << " now " << eventlist().now() << std::endl;
    if (ffapp->final_finish_time < finish_time) {
        ffapp->final_finish_time = finish_time;
    }

    for (uint64_t next_id: next_tasks) {
        FFTask * task = ffapp->tasks[next_id];
        task->counter--;
        std::cerr << this->taskid << " -> Task " << task->taskid << " counter at " << task->counter <<" name: "<<task->name <<" task type:"<< task->type<<std::endl;
        
        if (task->counter == 0) {
            task->ready_time = finish_time;
            task->state = FFTask::TASK_READY;
            if(task->type == FFTask::TASK_ALLTOALL && task->info.find("GROUP_BY forward") != std::string::npos && ffapp->is_mixnet && ffapp->topomanager){
                FFAlltoAll* all2all_task = dynamic_cast<FFAlltoAll*>(task);
                all2all_task->updatetrafficmatrix();
                // Trigger reconfiguration
                int region_id = (all2all_task->from_node_ids[0] / 8) / ffapp->topomanager->region_size;
                int layer_id = task->layer_id;
                int micro_batch_id = task->micro_batch_id;
                if (ffapp->topomanager->regional_topo_managers[region_id]->current_layer_id != layer_id || ffapp->topomanager->regional_topo_managers[region_id]->current_micro_batch_id != micro_batch_id) {
                    // trigger reconfiguration
                    //Deeksha: Add our reconfigutation and events here
                    ffapp->topomanager->regional_topo_managers[region_id]->current_layer_id = layer_id;
                    ffapp->topomanager->regional_topo_managers[region_id]->current_micro_batch_id = micro_batch_id;
                    // reset per-layer periodic reconf state
                    ffapp->topomanager->regional_topo_managers[region_id]->first_reconf_done = false;
                    ffapp->topomanager->regional_topo_managers[region_id]->conn_cycle_idx =0;
                    //ffapp->topomanager->demandrecorder->dump_all_traffic_matrices(layer_id, region_id, all2all_task->all2all_traffic_matrix);        //Deeksha: For debugging purpose
                    ffapp->topomanager->demandrecorder->append_traffic_matrix(layer_id, region_id, all2all_task->all2all_traffic_matrix);   //Deeksha: Occurs once per layer once per region
                    //Performing append (as above ) with normalized and scaled traffic matrix inside update traffic matrix function 
                    //ffapp->topomanager->demandrecorder->NormalizeTrafficMatrix(layer_id, region_id, all2all_task->all2all_traffic_matrix);
                    assert(ffapp->topomanager->demandrecorder->check_traffic_matrix(layer_id, region_id));// it must be true  //Deeksha: base will be true if append is done correctly, this function checks only the base
                    eventlist().sourceIsPending(*(ffapp->topomanager->regional_topo_managers[region_id]), eventlist().now());
                    //cout << "forward groupby task trigger reconfig at "<< eventlist().now() << std::endl;
                    //ffapp->topomanager->demandrecorder->print_traffic_matrix(layer_id, region_id);
                }
                else {
                    //Deeksha: Check Appendix B.2 in MixNet paper
                    assert(false && "first layer should always trigger reconfig"); // should not happen
                }
                eventlist().sourceIsPending(*task, task->ready_time + ffapp->topomanager->reconf_delay + 10);// add some delay for reconfig //Deeksha: +10 value hardcoded, not sure why? could be to avoid event overlap
            }
            else{
                eventlist().sourceIsPending(*task, task->ready_time);
            }
        }
    }
    if (ffapp->n_finished_tasks == ffapp->tasks.size()) {
        std::cerr << ffapp << " 0: finished one iter, nfin " << FFApplication::finished_apps << " ntot " << FFApplication::total_apps << std::endl;
        if (!ffapp->finished_once) {
            ffapp->finished_once = true;
            ffapp->first_iter_time = ffapp->final_finish_time;
            FFApplication::finished_apps++;
        }
        std::cerr << ffapp << " finished one iter, nfin " << FFApplication::finished_apps << " ntot " << FFApplication::total_apps << " now " << eventlist().now() << std::endl;
        if (FFApplication::finished_apps == FFApplication::total_apps) {
            eventlist().setEndtime(eventlist().now());
        }
        else {
            ffapp->reset_and_restart();
        }
    }
}

void FFTask::doNextEvent() {
    std::cerr << "FFTask::doNextEvent()"<<std::endl;
    std::cerr << "Task: " << this->taskid <<" name: "<<this->name<< " type: " << type << " counter: " << counter << std::endl;
    taskstart();
}

void FFTask::start_flow() {
    assert(false);//ff will never send flow
    // std::cerr << "task: " << (uint64_t)this << " start flow (" << src_node << ", " << dst_node << ")\n";
    start_time = ready_time;
    // // from ndp main application: generate flow

    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, eventlist(), src_node, dst_node, taskfinish, this);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(xfersize); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1000);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths = ffapp->topology->get_paths(src_node, dst_node);
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    choice = 0;
    vector<const Route*>* dstpaths = ffapp->topology->get_paths(dst_node, src_node);
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, start_time);

#ifdef PACKET_SCATTER 
    flowSrc->set_paths(srcpaths);
    flowSnk->set_paths(dstpaths);
#endif
    delete srcpaths;
    delete dstpaths;

}

// for comunication task
void taskfinish(void * task) {

    FFTask * fftask = static_cast<FFTask*>(task);
    // std::cerr << (uint64_t)task << " finished, calling back " <<std::endl;
    fftask->finish_time = fftask->eventlist().now();
    fftask->run_time = (fftask->finish_time - fftask->start_time);

    fftask->cleanup();
}

/* FFDevice */
FFDevice::FFDevice(FFApplication * ffapp, std::string type, float bandwidth, int node_id, int gpu_id,
                   int from_node, int to_node, int from_gpu, int to_gpu) {
    this->ffapp = ffapp;
    if (type == "DEVICE_GPU") {
        this->type = FFDeviceType::DEVICE_GPU;
    }
    else if (type == "DEVICE_CPU") {
        this->type = FFDeviceType::DEVICE_CPU;
    }
    else if (type == "DEVICE_GPU_COMM") {
        this->type = FFDeviceType::DEVICE_GPU_COMM;
    }
    else if (type == "DEVICE_DRAM_COMM") {
        this->type = FFDeviceType::DEVICE_DRAM_COMM;
    }
    else if (type == "DEVICE_NW_COMM") {
        this->type = FFDeviceType::DEVICE_NW_COMM;
    }
    else {
        throw "Unsupported device type!";
    }

    this->state = FFDevice::DEVICE_IDLE;
    this->bandwidth = bandwidth * 8 * 1000; 
    this->node_id = node_id;
    this->gpu_id = node_id;
    this->from_node = from_node;
    this->to_node = to_node;
    this->from_gpu = from_gpu;
    this->to_gpu = from_gpu;
    
    this->busy_up_to = 0;
}

FFDevice::FFDevice(FFApplication * ffapp, FlatBufTaskGraph::DeviceType devtype, uint64_t nodeid, 
             uint64_t deviceproperty, uint64_t bandwidth) {
    this->ffapp = ffapp;
    if (devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMP_GPU) {
        this->type = FFDeviceType::DEVICE_GPU;
        this->node_id = node_id;
        this->gpu_id = deviceproperty;
        this->from_node = 0;
        this->to_node = 0;
        this->from_gpu = 0;
        this->to_gpu = 0;
    }
    else if (devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMP_CPU) {
        this->type = FFDeviceType::DEVICE_CPU;
        this->node_id = node_id;
        this->gpu_id = deviceproperty;
        this->from_node = 0;
        this->to_node = 0;
        this->from_gpu = 0;
        this->to_gpu = 0;
    }
    else if (devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMM_NVLINK_COMM) {
        this->type = FFDeviceType::DEVICE_GPU_COMM;
        this->node_id = node_id;
        this->gpu_id = 0;
        this->from_node = 0;
        this->to_node = 0;
        this->from_gpu = deviceproperty / ffapp->ngpupernode;
        this->to_gpu = deviceproperty % ffapp->ngpupernode;
    }
    else if (devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_DEV_COMM 
          || devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_HOST_COMM) {
        this->type = FFDeviceType::DEVICE_DRAM_COMM;
        this->node_id = node_id;
        this->gpu_id = 0;
        this->from_node = 0;
        this->to_node = 0;
        this->from_gpu = 0;
        this->to_gpu = 0;
    }
    else if (devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_COMM || devtype == FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_NOMINAL) {
        this->type = FFDeviceType::DEVICE_NW_COMM;
        this->node_id = 0;
        this->gpu_id = 0;
        this->from_node = ffapp->gpus[deviceproperty / (ffapp->nnodes + ffapp->nswitches)];
        this->to_node = ffapp->gpus[deviceproperty % (ffapp->nnodes + ffapp->nswitches)];
        this->from_gpu = 0;
        this->to_gpu = 0;
    }
    else {
        throw "Unsupported device type!";
    }

    this->state = FFDevice::DEVICE_IDLE;
    this->bandwidth = bandwidth * 8 * 1000; 
    // this->node_id = node_id;
    // this->gpu_id = node_id;
    // this->from_node = from_node;
    // this->to_node = to_node;
    // this->from_gpu = from_gpu;
    // this->to_gpu = from_gpu;
    
    this->busy_up_to = 0;
}

#if 0

FFDevice::FFDevice(TaskGraphProtoBuf::Device_DeviceType type, 
                   float bandwidth, int node_id, int gpu_id,
                   int from_node, int to_node, int from_gpu, int to_gpu) {

    if (type == TaskGraphProtoBuf::Device_DeviceType_DEVICE_GPU) {
        this->type = FFDeviceType::DEVICE_GPU;
    }
    else if (type == TaskGraphProtoBuf::Device_DeviceType_DEVICE_CPU) {
        this->type = FFDeviceType::DEVICE_CPU;
    }
    else if (type == TaskGraphProtoBuf::Device_DeviceType_DEVICE_GPU_COMM) {
        this->type = FFDeviceType::DEVICE_GPU_COMM;
    }
    else if (type == TaskGraphProtoBuf::Device_DeviceType_DEVICE_DRAM_COMM) {
        this->type = FFDeviceType::DEVICE_DRAM_COMM;
    }
    else if (type == TaskGraphProtoBuf::Device_DeviceType_DEVICE_NW_COMM) {
        this->type = FFDeviceType::DEVICE_NW_COMM;
    }
    else {
        throw "Unsupported device type!";
    }

    this->state = FFDevice::DEVICE_IDLE;
    this->bandwidth = bandwidth * 8 * 1000; 
    this->node_id = node_id;
    this->gpu_id = node_id;
    this->from_node = from_node;
    this->to_node = to_node;
    this->from_gpu = from_gpu;
    this->to_gpu = from_gpu;
    
    this->busy_up_to = 0;
}

#endif

// FFRingAllReduce
FFRingAllreduce::FFRingAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_ALLREDUCE, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), 
    finished_curr_round(0), curr_round(0) {
    run_time = local_runtime * 1000000000ULL;
    operator_size = sz / ng.size() > 0 ? sz : ng.size();
    finished_rounds = std::vector<int>(ng.size(), 0);
}

void FFRingAllreduce::doNextEvent() {
    std::cerr << "FFRingAllreduce::doNextEvent()" << std::endl;
     std::cerr << "Task: " << this->taskid << " type: " << type << " counter: " << counter << std::endl;
    // this should only be called once...
    assert(curr_round == 0);

    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
        // std::cerr << "AR 1 node " << (uint64_t)this << " finished at " << this->finish_time << std::endl;
    }
    else {
        if (operator_size < DEFAULT_PACKET_SIZE * node_group.size() /* MTU */) {
            operator_size *= (2.0 * (node_group.size() - 1) / node_group.size());
        }
        finish_time = start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            start_flow(i, i);
            // start_flow(i, 0);
        }
    }

}
/*

original allreduce logic:
FFRingAllreduce::doNextEvent(): start the first round allreduce comm by calling start_flow(i,i)

at each start_flow(i,i): 
1. tcp connect->donextevent() 
2. after tcp send/recieve is done, call ar_finish_ring(f)

for each ar_finish_ring(f):
1. update allreduce task number
2. next round start_flow
*/
void FFRingAllreduce::start_flow(int src_idx, int id) {
    
    int src_gpu = ffapp->gpus[node_group[src_idx]];
    int dst_gpu = ffapp->gpus[node_group[(src_idx + 1) % node_group.size()]];

    FFRingAllreduceFlow * f = new FFRingAllreduceFlow();
    f->ar = this;
    f->id = id;
    f->src_idx = src_gpu;

    if ((src_gpu / NUM_GPU_PER_NODE) == (dst_gpu / NUM_GPU_PER_NODE)) { 
        //std::cerr << "intra-node communication" << std::endl;
        //intra-node communication
        finish_time = start_time + timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth);//refer to flexflow simulation
        std::cerr << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth) << " transmission size "<< operator_size << " group size" << node_group.size() << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        ar_finish_ring(f);
        return;
    }

    int choose_ocs=1;
    ofstream * fstream_out_;
    DCTCPSrc* flowSrc;
    if(ffapp->is_mixnet) {
        // we use the elec fabric by default
        flowSrc = new DCTCPSrc(NULL, NULL, ffapp->sub_fstream_out, eventlist(), src_gpu, dst_gpu, ar_finish_ring, f);
        flowSrc->is_all2all = false;
        flowSrc->is_elec = true;
    }
    else {
        flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, eventlist(), src_gpu, dst_gpu, ar_finish_ring, f);
    }
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size/node_group.size()); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);

    /*
        ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    */
    if (ffapp->tcpRtxScanner._enable_tcppairs) {
        tcp_info* flowinfo = new tcp_info(
                                    tcp_info::allreduce,
                                    micro_batch_id,
                                    layer_id,
                                    target_micro_batch_id,
                                    target_layer_id,
                                    id,
                                    operator_size
                                    );
        ffapp->tcpRtxScanner.registerTcp(*flowSrc, *flowinfo);
    }
    else {
        ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    }
    

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths;
    vector<const Route*>* dstpaths;

    // get path for allreduce traffic
    if(ffapp->is_mixnet) {
        srcpaths = ffapp->topology->get_eps_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_eps_paths(dst_gpu, src_gpu);
    }
    else {
        srcpaths = ffapp->topology->get_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_paths(dst_gpu, src_gpu);
    }

    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);
    choice = 0;
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round > 0 ? eventlist().now() : start_time + run_time);

#ifdef PACKET_SCATTER 
    flowSrc->set_paths(srcpaths);
    flowSnk->set_paths(dstpaths);
#endif
    delete srcpaths;
    delete dstpaths;

}

void ar_finish_ring(void * arinfo) {
    
    FFRingAllreduceFlow * f = static_cast<FFRingAllreduceFlow*>(arinfo);
    FFRingAllreduce * ar = f->ar;

    assert(ar->finished_rounds[f->id] == ar->curr_round);
    ar->finished_rounds[f->id]++; 
    ar->finished_curr_round++;
    delete f;
    if (ar->finished_curr_round == (int)ar->node_group.size()) {
        // ar->finished_partitions++;
        ar->curr_round++;
        ar->finished_curr_round = 0;
        if (ar->curr_round == 1 && (ar->operator_size / (2.0 * (ar->node_group.size() - 1) / ar->node_group.size())) <= DEFAULT_PACKET_SIZE * ar->node_group.size()) {
            std::cerr << "early terminiate..." << "ar->finish_time "<< ar->finish_time << " ar->eventlist().now() "<< ar->eventlist().now()<< std::endl;

            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
        }
        else if (ar->curr_round == 2 * ((int)ar->node_group.size() - 1)) {
            std::cerr << "finish first round " << "ar->finish_time "<< ar->finish_time << " ar->eventlist().now() "<< ar->eventlist().now()<< std::endl;
            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            // // std::cerr << "AR " << (uint64_t)ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
        else {
            ar->start_time = ar->finish_time;
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                ar->start_flow(i, i);
            }
        } 
    }

}




FFNewRingAllreduce::FFNewRingAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, const std::vector<std::vector<int>>& jumps, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime)
: FFTask(ffapp, FFTask::TASK_ALLREDUCE, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), jumps(jumps), operator_size(sz)
{
    run_time = local_runtime * 1000000000ULL;
    finished_curr_round = std::vector<int>(jumps.size(), 0);
    curr_round =  std::vector<int>(jumps.size(), 0);
    finished_rounds = std::vector<std::vector<int>>(jumps.size(), std::vector<int>(node_group.size(), 0));
    finished_rings = 0;
    for (auto j: jumps) {
        int sum_of_jump = 0;
        for (int k: j) sum_of_jump += k;
        total_jump.push_back(sum_of_jump);
    }
}

void FFNewRingAllreduce::doNextEvent() {
    std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    // std::cerr << "AR " << this << " size " <<  node_group.size() << " node starting " << jumps.size() << " rings, now " << eventlist().now() << std::endl;
    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
    }
    else {
        if (operator_size < DEFAULT_PACKET_SIZE  * node_group.size() /* MTU */) {
            operator_size *= (2.0 * (node_group.size() - 1) / node_group.size());
        }
        start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            for (size_t j = 0; j < jumps.size(); j++) {
                start_flow(i, jumps[j], j, i);
            }
        }
    }
}

void FFNewRingAllreduce::start_flow(int src_idx, const std::vector<int>& jump, int ring_id, int id)
{
    int src_node = ffapp->gpus[node_group[src_idx]];
    int dst_node = (src_idx + total_jump[ring_id]) % ffapp->nnodes; //ffapp->gpus[node_group[(src_idx + 1) % node_group.size()]];
    //std::cerr << "start flow src: " << src_node << " dst: " << dst_node << std::endl;

    FFNewRingAllreduceFlow * f = new FFNewRingAllreduceFlow();
    f->ar = this;
    f->id = id;
    f->ring_idx = ring_id;
    f->src_idx = src_idx;

    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_node, dst_node, ar_finish_newring, f);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size/node_group.size()/jumps.size()); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(10);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout = new Route();
    int curr = src_idx;
    for (int j: jump) {
        assert(static_cast<FlatTopology*>(ffapp->topology)->queues[ffapp->gpus[curr]][ffapp->gpus[(curr+j)%ffapp->nnodes]/*node_group.size()*/] != nullptr);
        routeout->push_back(static_cast<FlatTopology*>(ffapp->topology)->queues[ffapp->gpus[curr]][ffapp->gpus[(curr+j)%ffapp->nnodes]/*%node_group.size()*/]);
        routeout->push_back(static_cast<FlatTopology*>(ffapp->topology)->pipes[ffapp->gpus[curr]][ffapp->gpus[(curr+j)%ffapp->nnodes]/*%node_group.size()*/]);
        curr = (curr + j) % ffapp->nnodes /*% node_group.size()*/;
    }
    // assert(curr == (src_idx + total_jump[ring_id]) % ffapp->nnodes /*% node_group.size()*/);
    assert(ffapp->gpus[curr] == ffapp->gpus[(src_idx + total_jump[ring_id]) % ffapp->nnodes] /*% node_group.size()*/);
    routeout->push_back(flowSnk);

    Route* routein = new Route();
    curr = src_idx;
    for (int j: jump) {
        routein->push_front(static_cast<FlatTopology*>(ffapp->topology)->queues[ffapp->gpus[curr]][ffapp->gpus[(curr+j)%ffapp->nnodes]/*%node_group.size()*/]);
        routein->push_front(static_cast<FlatTopology*>(ffapp->topology)->pipes[ffapp->gpus[curr]][ffapp->gpus[(curr+j)%ffapp->nnodes]/*%node_group.size()*/]);
        curr = (curr + j) % ffapp->nnodes /*% node_group.size() */;
    }
    assert(ffapp->gpus[curr] == ffapp->gpus[(src_idx + total_jump[ring_id]) % ffapp->nnodes] /*% node_group.size()*/);
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round[ring_id] > 0 ? eventlist().now() : start_time + run_time);

#ifdef PACKET_SCATTER 
    flowSrc->set_paths(srcpaths);
    flowSnk->set_paths(dstpaths);
#endif

}

void ar_finish_newring(void * arinfo)
{
    FFNewRingAllreduceFlow * f = static_cast<FFNewRingAllreduceFlow*>(arinfo);
    FFNewRingAllreduce * ar = f->ar;
    int ring_idx = f->ring_idx;
    // std::cerr << "callback: ar " << f->ar << ", id: " << f->id << ", src_idx: " << f->src_idx << ", " << "ring_idx: " << f->ring_idx << ", round: " << f->round << std::endl;

    assert(ar->finished_rounds[f->ring_idx][f->id] == ar->curr_round[f->ring_idx]);
    ar->finished_rounds[f->ring_idx][f->id]++; 
    ar->finished_curr_round[f->ring_idx]++;
    delete f;
    // ar->total_finished_rounds++;

    if (ar->finished_curr_round[ring_idx] == (int)ar->node_group.size()) {
        // ar->finished_partitions++;
        ar->curr_round[ring_idx]++;
        ar->finished_curr_round[ring_idx] = 0;
        if (ar->curr_round[ring_idx] == 1 && (ar->operator_size / (2.0 * (ar->node_group.size() - 1) / ar->node_group.size())) <= DEFAULT_PACKET_SIZE * ar->node_group.size()) {
            std::cerr << "early terminiate..." << std::endl;
            ar->finished_rings++;
        }
        else if (ar->curr_round[ring_idx] == 2 * ((int)ar->node_group.size() - 1)) {
            ar->finished_rings++;
        }
        else {
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                ar->start_flow(i, ar->jumps[ring_idx], ring_idx, i);
            }
        }
        // std::cerr << ar << ": ring finished " << ar->finished_rings << std::endl;
        if (ar->finished_rings == ar->jumps.size()) {
            ar->finish_time = ar->eventlist().now();
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            std::cerr << "AR " << ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
         
    }
}

// PARAMETER SERVER
FFPSAllreduce::FFPSAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, /*int pserver,*/std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_ALLREDUCE, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), operator_size(sz),
    /*pserver(pserver),*/ curr_round(0) , finished_curr_round(0)
{
    run_time = local_runtime * 1000000000ULL;
    pserver = ng[0];
    finished_rounds = std::vector<int>(ng.size(), 0);
}

void FFPSAllreduce::doNextEvent() {
    std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    // this should only be called once...
    assert(curr_round == 0);
    

    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
        // std::cerr << "AR 1 node " << (uint64_t)this << " finished at " << this->finish_time << std::endl;
    }
    else {
        start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            if (node_group[i] != pserver)
                start_flow(i, 0);
        }
    }
}

void FFPSAllreduce::start_flow(int node_idx, int direction) {

    assert(direction < 2);

    int src_node, dst_node;
    if (direction == 0) {
        src_node = ffapp->gpus[node_group[node_idx]];
        dst_node = ffapp->gpus[pserver];
    }
    else {
        dst_node = ffapp->gpus[node_group[node_idx]];
        src_node = ffapp->gpus[pserver];
    }

    // std::cerr << "AR task: " << (uint64_t)this << " start flow (" << src_node << ", " << dst_node << ") round " << curr_round << " nsize " << node_group.size() << " pserver " << pserver << "\n";

    FFPSAllreduceFlow * f = new FFPSAllreduceFlow();
    f->ar = this;
    f->node_idx = node_idx;
    f->direction = direction;

    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_node, dst_node, ar_finish_ps, f);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths = ffapp->topology->get_paths(src_node, dst_node);
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    choice = 0;
    vector<const Route*>* dstpaths = ffapp->topology->get_paths(dst_node, src_node);
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round > 0 ? eventlist().now() : start_time);

#ifdef PACKET_SCATTER 
    flowSrc->set_paths(srcpaths);
    flowSnk->set_paths(dstpaths);
#endif
    delete srcpaths;
    delete dstpaths;
}

void ar_finish_ps(void * arinfo) {
    
    FFPSAllreduceFlow * f = static_cast<FFPSAllreduceFlow*>(arinfo);
    FFPSAllreduce * ar = f->ar;

    assert(ar->finished_rounds[f->node_idx] == ar->curr_round);
    ar->finished_rounds[f->node_idx]++; 
    ar->finished_curr_round++;
    delete f;

    if (ar->finished_curr_round == (int)ar->node_group.size() - 1) {
        // ar->finished_partitions++;
        ar->curr_round++;
        ar->finished_curr_round = 0;
        if (ar->curr_round == 2) {
            ar->finish_time = ar->eventlist().now();            
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            // // std::cerr << "AR " << (uint64_t)ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
        else {
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                if (ar->pserver != ar->node_group[i])
                    ar->start_flow(i, 1);
            }
        } 
    }
}


FFDPSAllreduce::FFDPSAllreduce(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_ALLREDUCE, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), 
    finished_curr_round(0), curr_round(0) {
    run_time = local_runtime * 1000000000ULL;
    operator_size = sz / ng.size() > 0 ? sz : ng.size();
    // finished_rounds = std::vector<int>(ng.size(), 0);
}

void FFDPSAllreduce::doNextEvent() {
    std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    // this should only be called once...
    assert(curr_round == 0);
    

    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
        // std::cerr << "AR 1 node " << (uint64_t)this << " finished at " << this->finish_time << std::endl;
    }
    else {
        start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            for (size_t j = 0; j < node_group.size(); j++) {
                if (i != j)
                    start_flow(i, j);
            }
        }
    }

}

void FFDPSAllreduce::start_flow(int src_node, int dst_node) {
   
    // std::cerr << "AR task: " << (uint64_t)this << " start flow (" << src_node << ", " << dst_node << ") round " << curr_round << " nsize " << node_group.size() << "\n";

    // f->src_idx = src_idx;
    // f->round = round;
    src_node = ffapp->gpus[node_group[src_node]];
    dst_node = ffapp->gpus[node_group[dst_node]];

    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_node, dst_node, ar_finish_dps, this);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size/node_group.size()); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths = ffapp->topology->get_paths(src_node, dst_node);
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    choice = 0;
    vector<const Route*>* dstpaths = ffapp->topology->get_paths(dst_node, src_node);
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round > 0 ? eventlist().now() : start_time);

#ifdef PACKET_SCATTER 
    flowSrc->set_paths(srcpaths);
    flowSnk->set_paths(dstpaths);
#endif
    delete srcpaths;
    delete dstpaths;

}

void ar_finish_dps(void * ar_ptr) {
    
    FFDPSAllreduce * ar = static_cast<FFDPSAllreduce*>(ar_ptr);

    ar->finished_curr_round++;

    if (ar->finished_curr_round == (int)(ar->node_group.size() * (ar->node_group.size() - 1))) {
        // ar->finished_partitions++;
        ar->curr_round++;
        ar->finished_curr_round = 0;
        if (ar->curr_round == 2) {
            ar->finish_time = ar->eventlist().now();
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            // // std::cerr << "AR " << (uint64_t)ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
        else {
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                for (size_t j = 0; j < ar->node_group.size(); j++) {
                    if (i != j)
                        ar->start_flow(i, j);
                }
            }
        } 
    }
}

// FFReduceScatter
FFReduceScatter::FFReduceScatter(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_REDUCESCATTER, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), 
    finished_curr_round(0), curr_round(0) {
    run_time = local_runtime * 1000000000ULL;
    operator_size = sz / ng.size() > 0 ? sz : ng.size();
    finished_rounds = std::vector<int>(ng.size(), 0);
}

void FFReduceScatter::doNextEvent() {
    std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    // this should only be called once...
    assert(curr_round == 0);

    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
        // std::cerr << "AR 1 node " << (uint64_t)this << " finished at " << this->finish_time << std::endl;
    }
    else {
        if (operator_size < DEFAULT_PACKET_SIZE * node_group.size() /* MTU */) {
            operator_size *= ((node_group.size() - 1) / node_group.size());
        }
        start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            start_flow(i, i);
        }
    }

}

void FFReduceScatter::start_flow(int src_idx, int id) {
   
    // std::cerr << "AR task: " << (uint64_t)this << " start flow (" << src_node << ", " << dst_node << ") round " << curr_round << " nsize " << node_group.size() << "\n";

    // f->src_idx = src_idx;
    // f->round = round;
    src_node = ffapp->gpus[node_group[src_idx]];
    dst_node = ffapp->gpus[node_group[(src_idx + 1) % node_group.size()]];

    FFReduceScatterFlow * f = new FFReduceScatterFlow();
    f->rs = this;
    f->id = id;

    if ((src_node / NUM_GPU_PER_NODE) == (dst_node / NUM_GPU_PER_NODE)) { 
        //std::cerr << "intra-node communication" << std::endl;
        //intra-node communication
        finish_time = start_time + timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth);//refer to flexflow simulation
        std::cerr << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth) << " transmission size "<< operator_size << " group size" << node_group.size() << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        ar_finish_reducescatter(f);
        return;
    }

    // Transform src_gpu to src_node, dst_gpu to dst_node
    int src_gpu = src_node % NUM_GPU_PER_NODE;
    int dst_gpu = dst_node % NUM_GPU_PER_NODE;
    if (src_gpu != dst_gpu) {
        // add intra-node communication time
        finish_time += timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth);
    }
    src_node /= NUM_GPU_PER_NODE;
    dst_node /= NUM_GPU_PER_NODE;

    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_node, dst_node, ar_finish_reducescatter, this);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size/node_group.size()); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths = ffapp->topology->get_paths(src_node, dst_node);
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    choice = 0;
    vector<const Route*>* dstpaths = ffapp->topology->get_paths(dst_node, src_node);
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round > 0 ? eventlist().now() : start_time);

    delete srcpaths;
    delete dstpaths;
}

void ar_finish_reducescatter(void * ar_ptr) {
    FFReduceScatterFlow * f = static_cast<FFReduceScatterFlow*>(ar_ptr);
    FFReduceScatter * ar = f->rs;

    assert(ar->finished_rounds[f->id] == ar->curr_round);
    ar->finished_rounds[f->id]++;
    ar->finished_curr_round++;
    delete f;
    if (ar->finished_curr_round == (int)(ar->node_group.size())) {
        // ar->finished_partitions++;
        ar->curr_round++;
        ar->finished_curr_round = 0;
        if (ar->curr_round == 1 && (ar->operator_size / ((ar->node_group.size() - 1) / ar->node_group.size())) <= DEFAULT_PACKET_SIZE * ar->node_group.size()) {
            std::cerr << "early terminiate..." << std::endl;
            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
        }
        else if (ar->curr_round == ((int)ar->node_group.size() - 1)) {
            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            // // std::cerr << "AR " << (uint64_t)ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
        else {
            ar->start_time = ar->finish_time;
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                ar->start_flow(i, i);
            }
        } 
    }
}

// FFAllGather
FFAllGather::FFAllGather(FFApplication * ffapp, uint64_t taskid, std::vector<uint64_t> ng, int counter, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_ALLGATHER, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), node_group(ng), 
    finished_curr_round(0), curr_round(0) {
    run_time = local_runtime * 1000000000ULL;
    operator_size = sz / ng.size() > 0 ? sz : ng.size();
    finished_rounds = std::vector<int>(ng.size(), 0);
}

void FFAllGather::doNextEvent() {
    std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    // this should only be called once...
    assert(curr_round == 0);

    if (node_group.size() == 1) {
        // finished_partitions = 1;
        finish_time = start_time = ready_time;
        // state = FFTask::TASK_FINISHED;
        cleanup();
        // std::cerr << "AR 1 node " << (uint64_t)this << " finished at " << this->finish_time << std::endl;
    }
    else {
        if (operator_size < DEFAULT_PACKET_SIZE * node_group.size() /* MTU */) {
            operator_size *= ((node_group.size() - 1) / node_group.size());
        }
        start_time = ready_time;
        state = FFTask::TASK_RUNNING;
        for (size_t i = 0; i < node_group.size(); i++) {
            start_flow(i, i);
        }
    }

}

void FFAllGather::start_flow(int src_idx, int id) {
   
    // std::cerr << "AR task: " << (uint64_t)this << " start flow (" << src_node << ", " << dst_node << ") round " << curr_round << " nsize " << node_group.size() << "\n";

    // f->src_idx = src_idx;
    // f->round = round;
    src_node = ffapp->gpus[node_group[src_idx]];
    dst_node = ffapp->gpus[node_group[(src_idx + 1) % node_group.size()]];

    FFAllGatherFlow * f = new FFAllGatherFlow();
    f->ag = this;
    f->id = id;

    if ((src_node / NUM_GPU_PER_NODE) == (dst_node / NUM_GPU_PER_NODE)) { 
        //std::cerr << "intra-node communication" << std::endl;
        //intra-node communication
        finish_time = start_time + timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth);//refer to flexflow simulation
        std::cerr << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth) << " transmission size "<< operator_size << " group size" << node_group.size() << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        ar_finish_allgather(f);
        return;
    }

    // Transform src_gpu to src_node, dst_gpu to dst_node
    int src_gpu = src_node % NUM_GPU_PER_NODE;
    int dst_gpu = dst_node % NUM_GPU_PER_NODE;
    if (src_gpu != dst_gpu) {
        // add intra-node communication time
        finish_time += timeFromSec((operator_size / node_group.size()) / ffapp->nvlink_bandwidth);
    }
    src_node /= NUM_GPU_PER_NODE;
    dst_node /= NUM_GPU_PER_NODE;


    DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_node, dst_node, ar_finish_allgather, this);
    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size/node_group.size()); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);
    
    ffapp->tcpRtxScanner.registerTcp(*flowSrc);

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths = ffapp->topology->get_paths(src_node, dst_node);
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    choice = 0;
    vector<const Route*>* dstpaths = ffapp->topology->get_paths(dst_node, src_node);
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, 
        curr_round > 0 ? eventlist().now() : start_time);

    delete srcpaths;
    delete dstpaths;
}

void ar_finish_allgather(void * ar_ptr) {
    
    FFAllGatherFlow * f = static_cast<FFAllGatherFlow*>(ar_ptr);
    FFAllGather * ar = f->ag;

    assert(ar->finished_rounds[f->id] == ar->curr_round);
    ar->finished_rounds[f->id]++; 
    ar->finished_curr_round++;
    delete f;
    if (ar->finished_curr_round == (int)(ar->node_group.size() - 1)) {
        // ar->finished_partitions++;
        ar->curr_round++;
        ar->finished_curr_round = 0;
        if (ar->curr_round == 1 && (ar->operator_size / ((ar->node_group.size() - 1) / ar->node_group.size())) <= DEFAULT_PACKET_SIZE * ar->node_group.size()) {
            std::cerr << "early terminiate..." << std::endl;
            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
        }
        else if (ar->curr_round == ((int)ar->node_group.size() - 1)) {
            ar->finish_time = max(ar->finish_time, ar->eventlist().now());
            ar->cleanup();
            // ar->state = FFTask::TASK_FINISHED;
            // // std::cerr << "AR " << (uint64_t)ar << " finished at " << ar->finish_time << std::endl;
            // ar->ffapp->n_finished_tasks++;
            // if (ar->ffapp->final_finish_time < ar->finish_time) {
            //     ar->ffapp->final_finish_time = ar->finish_time;
            // }
        }
        else {
            ar->start_time = ar->finish_time;
            for (size_t i = 0; i < ar->node_group.size(); i++) {
                ar->start_flow(i, i);
            }
        } 
    }
}

// FFAlltoAll
FFAlltoAll::FFAlltoAll(FFApplication * ffapp, uint64_t taskid, int counter, std::vector<uint64_t> fn, std::vector<uint64_t> tn, std::unordered_map<std::pair<int, int>, uint64_t, pair_hash> o_sizes, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime) :
    FFTask(ffapp, FFTask::TASK_ALLTOALL, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id), from_node_ids(fn), to_node_ids(tn), operator_sizes(o_sizes), finished_partitions(0), curr_round(0), total_rounds(0), all2all_traffic_matrix(ffapp->ep_degree*ffapp->tp_degree/8, ffapp->ep_degree*ffapp->tp_degree/8) {
    // we assume each node have 8 gpus
    run_time = local_runtime * 1000000000ULL;
    // Initialize traffic statistics maps
    node_total_volume.clear();
    node_ocs_volume.clear();
    node_eps_volume.clear();
    node_total_recv_volume.clear();
    node_ocs_recv_volume.clear();
    node_eps_recv_volume.clear();
}

void FFAlltoAll::updatetrafficmatrix() {
    if (flag ==1){
        return;
    }
    for (size_t i = 0; i < from_node_ids.size()/ffapp->tp_degree; i++) {
        for (size_t j = 0; j < to_node_ids.size()/ffapp->tp_degree; j++) {
            for(int tp_idx=0;tp_idx<ffapp->tp_degree;tp_idx++) {
                int flow_size = operator_sizes[std::make_pair(from_node_ids[i*ffapp->tp_degree+tp_idx], to_node_ids[j*ffapp->tp_degree+tp_idx])];
                int src_node = from_node_ids[i*ffapp->tp_degree+tp_idx] / 8;
                int dst_node = to_node_ids[j*ffapp->tp_degree+tp_idx] / 8;
                int region_size = ffapp->topomanager->region_size;
                all2all_traffic_matrix.add_elem_by(src_node%region_size, dst_node%region_size, flow_size);
            }
        }
    }
    flag=0;
}

void FFAlltoAll::doNextEvent() {
    //std::cerr << "Task: " << (uint64_t)this << " type: " << type << " counter: " << counter << std::endl;
    std::cerr << "FFAlltoAll::doNextEvent()" <<std::endl;
    std::cerr << "Task: " << this->taskid <<" name: "<<this->name<< " type: " << type << " counter: " << counter << std::endl;
    assert(curr_round == 0);
    assert(total_rounds == 0);
    start_time = ready_time;
    state = FFTask::TASK_RUNNING;
    // calculate the total rounds
    for (size_t i = 0; i < from_node_ids.size()/ffapp->tp_degree; i++) {
        for (size_t j = 0; j < to_node_ids.size()/ffapp->tp_degree; j++) {
            for(int tp_idx=0;tp_idx<ffapp->tp_degree;tp_idx++) {
                total_rounds++;
            }
        }
    }
    
    std::cerr<< " all2all task " << " total rounds " << total_rounds <<" from node id size: " << from_node_ids.size() << " to node id size: " << to_node_ids.size() << std::endl;

    int region_size = ffapp->topomanager ? ffapp->topomanager->region_size : 0;
    std::vector<std::vector<uint64_t>> region_matrix(
        region_size, std::vector<uint64_t>(region_size, 0));
    
    // Start flows
    for (size_t i = 0; i < from_node_ids.size()/ffapp->tp_degree; i++) {
        for (size_t j = 0; j < to_node_ids.size()/ffapp->tp_degree; j++) {
            for(int tp_idx=0;tp_idx<ffapp->tp_degree;tp_idx++) {
                int src_gpu = from_node_ids[i*ffapp->tp_degree+tp_idx];
                int dst_gpu = to_node_ids[j*ffapp->tp_degree+tp_idx];
                start_flow(src_gpu, dst_gpu);
                uint64_t flow_size = operator_sizes[std::make_pair(src_gpu, dst_gpu)];
                int src_node = src_gpu / NUM_GPU_PER_NODE;
                int dst_node = dst_gpu / NUM_GPU_PER_NODE;
                int src_region = region_size > 0 ? src_node % region_size : -1;
                int dst_region = region_size > 0 ? dst_node % region_size : -1;
                if (region_size > 0 && src_node != dst_node) {
                    region_matrix[src_region][dst_region] += flow_size;
                }
                std::cout << "A2A_FLOW task_id=" << this->taskid
                          << " micro_batch=" << this->micro_batch_id
                          << " layer=" << this->layer_id
                          << " src_gpu=" << src_gpu
                          << " dst_gpu=" << dst_gpu
                          << " src_node=" << src_node
                          << " dst_node=" << dst_node
                          << " src_region=" << src_region
                          << " dst_region=" << dst_region
                          << " flow_size=" << flow_size
                          << std::endl;
            }
        }
    }

    if (region_size > 0) {
        std::cout << "A2A_REGION_MATRIX_BEGIN task_id=" << this->taskid
                  << " micro_batch=" << this->micro_batch_id
                  << " layer=" << this->layer_id
                  << " region_size=" << region_size
                  << std::endl;
        dump_region_matrix(std::cout, region_matrix);
        std::cout << "A2A_REGION_MATRIX_END task_id=" << this->taskid
                  << " micro_batch=" << this->micro_batch_id
                  << " layer=" << this->layer_id
                  << std::endl;
    }
}

void FFAlltoAll::start_flow(int src_gpu, int dst_gpu) {
    /*
        mixnet:
            1. conn is node matrix not expert matrix
            2. we need to set the is_all2all and is_elec to facilitate the reconfig
            3. to consistent with other topology, the src/dst of flow should be gpu id we convert it inside the node, it only affect the logger info
    */
    
    uint64_t flow_size;
    
    flow_size = operator_sizes[std::make_pair(src_gpu, dst_gpu)];
    std::cerr << " FFAlltoAll::start_flow: "<<" from "<< src_gpu << " to "<< dst_gpu <<" size " << flow_size <<" task id: " << this->taskid << std::endl;
    // f->src_idx = src_idx;
    // f->round = round;
    assert(ffapp->gpus[src_gpu]==src_gpu && ffapp->gpus[dst_gpu]==dst_gpu);

    FFAlltoAllFlow * f = new FFAlltoAllFlow();
    f->a2a = this;
    f->src_idx = src_gpu;
    f->dst_idx = dst_gpu;

    if ((src_gpu / NUM_GPU_PER_NODE) == (dst_gpu / NUM_GPU_PER_NODE)) {
        // intra-node communication
        finish_time = start_time + timeFromSec(flow_size / ffapp->nvlink_bandwidth);
        std::cerr << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec(flow_size / ffapp->nvlink_bandwidth) << " transmission size "<< flow_size << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        f->intra_node_routing++;
        //Deeksha: Debugging
        //std::cout << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec(flow_size / ffapp->nvlink_bandwidth) << " transmission size "<< flow_size << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        finish_alltoall(f);
        return;
    }


    DCTCPSrc* flowSrc;
    int is_ocs = 0;//elec by default
    flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, eventlist(), src_gpu, dst_gpu,finish_alltoall , f); // all2all flow in ocs log
    if(ffapp->is_mixnet) {
        int src_node = src_gpu / NUM_GPU_PER_NODE;
        int dst_node = dst_gpu / NUM_GPU_PER_NODE;
        flowSrc->is_all2all = true;
        if(ffapp->topology->conn[src_node][dst_node]>0) {//ocs
            // set the iselec and isall2all
            flowSrc->is_elec = false; //Deeksha: All electric works well getting back to ocs for debugging
            // Update OCS volume statistics (sent)
            node_ocs_volume[src_node] += flow_size;
            // Update OCS volume statistics (received)
            node_ocs_recv_volume[dst_node] += flow_size;
            std::cerr << "flow goes through OCS, update OCS volume statistics: src_node " << src_gpu << " dst_node " << dst_gpu << " flow_size " << flow_size << "flow id " << this->taskid << std::endl;
        }
        else {
            //Add tje flow to ocs but pause the flow immediately, we will resume the flow after reconfig
            flowSrc->is_elec = false; //Deeksha: All electric works well getting back to ocs for debugging
            // Update OCS volume statistics (sent)
            node_ocs_volume[src_node] += flow_size;
            // Update OCS volume statistics (received)
            node_ocs_recv_volume[dst_node] += flow_size;
                std::cerr << "flow goes through OCS, update OCS volume statistics: src_node " << src_gpu << " dst_node " << dst_gpu << " flow_size " << flow_size << "flow id " << this->taskid << std::endl;
            // flowSrc->is_elec = true;
            // // Update EPS volume statistics (sent)
            // node_eps_volume[src_node] += flow_size;
            // // Update EPS volume statistics (received)
            // node_eps_recv_volume[dst_node] += flow_size;
            //     std::cerr << "flow goes through EPS, update EPS volume statistics: src_node " << src_gpu << " dst_node " << dst_gpu << " flow_size " << flow_size << "flow id " << this->taskid << std::endl;
        }
        // Update total volume statistics (sent)
        node_total_volume[src_node] += flow_size;
        // Update total volume statistics (received)
        node_total_recv_volume[dst_node] += flow_size;
    }

    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(flow_size); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1000);
    
    // ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    if (ffapp->tcpRtxScanner._enable_tcppairs) {
        tcp_info* flowinfo = new tcp_info(
                                    info,
                                    micro_batch_id,
                                    layer_id,
                                    target_micro_batch_id,
                                    target_layer_id,
                                    0,
                                    flow_size
                                    );
        ffapp->tcpRtxScanner.registerTcp(*flowSrc, *flowinfo);
    }
    else {
        ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    }

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths;
    vector<const Route*>* dstpaths;
    if(flowSrc->is_elec) {
        assert(ffapp->is_mixnet);
        srcpaths = ffapp->topology->get_eps_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_eps_paths(dst_gpu, src_gpu);
    }
    else {
        srcpaths = ffapp->topology->get_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_paths(dst_gpu, src_gpu);
    }

    // set routeout
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);

    // set routein
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    /*
        if this topo is mixnet and this flow go ocs: we need to check the status of connection
            if the this regional ocs is doing reconfig, we need to pause the flow
            otherwise, we can start the flow immediately
    */
    if(ffapp->is_mixnet) {
        int region_id = (src_gpu/8) / ffapp->topomanager->region_size;
        assert(region_id == (dst_gpu/8) / ffapp->topomanager->region_size);
        int is_doing_reconfig = (ffapp->topomanager->regional_topo_managers[region_id]->status == RegionalTopoManager::TopoStatus::TOPO_RECONF);
        if(is_doing_reconfig && flowSrc->is_elec == false) {
            simtime_picosec start_time_;
            if (curr_round == 0) {
                start_time_ = std::max(ffapp->topomanager->regional_topo_managers[region_id]->reconfig_end_time, start_time);
            } else {
                start_time_ = ffapp->topomanager->regional_topo_managers[region_id]->reconfig_end_time;
            }
            flowSrc->connect(*routeout, *routein, *flowSnk, start_time_);
        }
        else {
            flowSrc->connect(*routeout, *routein, *flowSnk, curr_round > 0 ? eventlist().now() : start_time);
        }
        // vamsi: Let the tcp send everything out at one go, and let the topology deal with it.
        flowSrc->_cwnd = 0xffffffff;
    }
    else {
        flowSrc->connect(*routeout, *routein, *flowSnk, curr_round > 0 ? eventlist().now() : start_time);
    }

    delete srcpaths;
    delete dstpaths;
}

void finish_alltoall(void * a2ainfo) {
    FFAlltoAllFlow * a2a = static_cast<FFAlltoAllFlow*>(a2ainfo);
    FFAlltoAll * a2a_task = a2a->a2a;
    
    std::cerr << "finish_alltoall "<< a2a_task->id <<" from " << a2a->src_idx << " to " << a2a->dst_idx << std::endl;

    a2a_task->curr_round++;
    std::cerr << "alltoall round_complete task_id " << a2a_task->taskid
              << " layer " << a2a_task->layer_id
              << " micro_batch " << a2a_task->micro_batch_id
              << " round " << a2a_task->curr_round << "/" << a2a_task->total_rounds
              << " time " << a2a_task->eventlist().now() << std::endl;
    
    if (a2a_task->curr_round == a2a_task->total_rounds) {
        std::cerr << "finish_alltoall_task "<< " total rounds "<< a2a_task->total_rounds << " intra node routing num "<< a2a->intra_node_routing << std::endl;
        a2a_task->finish_time = max(a2a_task->finish_time, a2a_task->eventlist().now());
        a2a_task->run_time = a2a_task->finish_time - a2a_task->start_time;
        std::cerr << "alltoall timing task_id " << a2a_task->taskid
                  << " layer " << a2a_task->layer_id
                  << " micro_batch " << a2a_task->micro_batch_id
                  << " start " << a2a_task->start_time
                  << " finish " << a2a_task->finish_time
                  << " duration " << a2a_task->run_time << std::endl;
        // vamsi: toggle the direction, so that the topology is reverse for next phase of all to all
        int region_id = (a2a_task->from_node_ids[0] / 8) / a2a_task->ffapp->topomanager->region_size;
        a2a_task->ffapp->topomanager ? a2a_task->ffapp->topomanager->transpose(region_id) : void();
        a2a_task->cleanup();
        // if (FFApplication::debug_exit==a2a_task->layer_id) {
        //     exit(0);
        // }
    }
}

FFP2P::FFP2P(FFApplication * ffapp, uint64_t taskid, int counter, uint64_t src_idx, uint64_t dst_idx, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime)
: FFP2P(ffapp, taskid, counter, std::vector<uint64_t>{src_idx}, std::vector<uint64_t>{dst_idx}, sz, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id, local_runtime) {
}

FFP2P::FFP2P(FFApplication * ffapp, uint64_t taskid, int counter, std::vector<uint64_t> src_indices_, std::vector<uint64_t> dst_indices_, uint64_t sz, std::string info, int micro_batch_id, int layer_id, int target_micro_batch_id, int target_layer_id, double local_runtime)
: FFTask(ffapp, FFTask::TASK_P2P, taskid, counter, info, micro_batch_id, layer_id, target_micro_batch_id, target_layer_id),
  src_indices(std::move(src_indices_)),
  dst_indices(std::move(dst_indices_)),
  operator_size(sz) {
    assert(src_indices.size() == dst_indices.size());
    total_flows = src_indices.size();
    finished_flows = 0;
    run_time = local_runtime * 1000000000ULL;
}

void FFP2P::doNextEvent() {
    std::cerr << "FFP2P::doNextEvent()" <<std::endl;
    std::cerr << "Task: " << this->taskid <<" name: "<<this->name<< " type: " << type << " counter: " << counter << std::endl;
    start_time = ready_time;
    state = FFTask::TASK_RUNNING;
    finish_time = start_time;
    finished_flows = 0;

    if (total_flows == 0) {
        cleanup();
        return;
    }

    for (size_t i = 0; i < total_flows; ++i) {
        start_flow(src_indices[i], dst_indices[i]);
    }
}

void FFP2P::start_flow(uint64_t src_gpu, uint64_t dst_gpu) {
    assert(ffapp->gpus[src_gpu]==src_gpu && ffapp->gpus[dst_gpu]==dst_gpu);

    FFP2PFlow * flow = new FFP2PFlow();
    flow->p2p = this;
    flow->src_idx = src_gpu;
    flow->dst_idx = dst_gpu;

    if ((src_gpu / NUM_GPU_PER_NODE) == (dst_gpu / NUM_GPU_PER_NODE)) {
        // intra-node communication
        simtime_picosec done_time = start_time + timeFromSec(operator_size / ffapp->nvlink_bandwidth);//refer to flexflow simulation
        finish_time = std::max(finish_time, done_time);
        std::cerr << "intra-node communication" << " start time: "<< start_time <<" transmission time: " << timeFromSec(operator_size / ffapp->nvlink_bandwidth) << " transmission size "<< operator_size << " nvlink bandwidth" << ffapp->nvlink_bandwidth << " finish time "<< finish_time << std::endl;
        finish_p2p(flow);
        return;
    }
    DCTCPSrc* flowSrc;
    
    if(ffapp->is_mixnet) {
        flowSrc = new DCTCPSrc(NULL, NULL, ffapp->sub_fstream_out, eventlist(), src_gpu, dst_gpu, finish_p2p, flow);
    }
    else {
        flowSrc = new DCTCPSrc(NULL, NULL, ffapp->fstream_out, 
        eventlist(), src_gpu, dst_gpu, finish_p2p, flow);
    }

    TcpSink* flowSnk = new TcpSink();
    flowSrc->set_flowsize(operator_size); // bytes
    flowSrc->set_ssthresh(ffapp->ssthresh*Packet::data_packet_size());
    flowSrc->_rto = timeFromMs(1);
    
    //ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    if (ffapp->tcpRtxScanner._enable_tcppairs){
        tcp_info* flowinfo = new tcp_info(
                                    tcp_info::p2p,
                                    micro_batch_id,
                                    layer_id,
                                    target_micro_batch_id,
                                    target_layer_id,
                                    0,
                                    operator_size
                                    );
        ffapp->tcpRtxScanner.registerTcp(*flowSrc, *flowinfo);
    }
    else {
        ffapp->tcpRtxScanner.registerTcp(*flowSrc);
    }

    Route* routeout, *routein;

    int choice = 0;
    vector<const Route*>* srcpaths;
    vector<const Route*>* dstpaths;
    if(ffapp->is_mixnet) {
        flowSrc->is_elec = true;
        flowSrc->is_all2all = false;
        srcpaths = ffapp->topology->get_eps_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_eps_paths(dst_gpu, src_gpu);
    }
    else {
        srcpaths = ffapp->topology->get_paths(src_gpu, dst_gpu);
        dstpaths = ffapp->topology->get_paths(dst_gpu, src_gpu);
    }
    // set routeout
    choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
    routeout = new Route(*(srcpaths->at(choice)));
    routeout->push_back(flowSnk);
    //set routein
    choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
    routein = new Route(*(dstpaths->at(choice)));
    routein->push_back(flowSrc);

    flowSrc->connect(*routeout, *routein, *flowSnk, start_time);

    delete srcpaths;
    delete dstpaths;
}

void finish_p2p(void * p2pinfo) {
    FFP2PFlow * flow = static_cast<FFP2PFlow*>(p2pinfo);
    FFP2P * p2p = flow->p2p;

    p2p->finish_time = max(p2p->finish_time, p2p->eventlist().now());

    delete flow;

    p2p->finished_flows++;
    if (p2p->finished_flows == p2p->total_flows) {
        p2p->run_time = (p2p->finish_time - p2p->start_time);
        p2p->cleanup();
    }
}
