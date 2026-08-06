#ifndef VERMILION_TOPO_MANAGER_H
#define VERMILION_TOPO_MANAGER_H
// #ifndef FLAT_TOPO
// #define FLAT_TOPO
#include <iostream>
#include <unordered_map>
#include "main.h"
#include "randomqueue.h"
#include <queue>
#include "pipe.h"
#include "config.h"
#include "loggers.h"
#include "network.h"
#include "firstfit.h"
#include "topology.h"
#include "logfile.h"
#include "eventlist.h"
#include "switch.h"
#include "vermilion.h"
#include <algorithm>
#include <random>
#include <cstdint>
#include "dyn_net_sch.h"
#include <numeric> 
#include "ecnqueue.h"
#include "queue_lossless.h"
#include "tcp.h"
#ifndef QT
#define QT
typedef enum {RANDOM, ECN, COMPOSITE, CTRL_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN} queue_type;
#endif
using namespace std;


class TcpRtxTimerScanner;

enum VermilionConnCycleAlgorithm {
  CONN_CYCLE_VERMILION = 0,
  CONN_CYCLE_ROTOR = 1,
  CONN_CYCLE_BIRKHOFF = 2
};

struct All2AllTrafficRecorder {
  All2AllTrafficRecorder(int layer_num, int region_num, int region_size, TcpRtxTimerScanner * rtx_scanner);
  void append_traffic_matrix(int layer_id, int region_id, Matrix2D<double> & tm);
  void NormalizeTrafficMatrix(int layer_id, int region_id, Matrix2D<double> &tm); //Deeksha: Declared NormalizeTrafficMatrix
  void VermilionGraph(int layer_id, int region_id, Matrix2D<double> &tm); //Deeksha: Declared VermilionGraph
  bool check_traffic_matrix(int layer_id, int region_id);
  void dump_all_traffic_matrices(int layer_id, int region_id, Matrix2D<double> & tm) const;
  TcpRtxTimerScanner * rtx_scanner;
  std::vector<std::vector<Matrix2D<double>* >> traffic_matrix; // [layer, region, matrix]
  int layer_num;
  int region_num;
  int region_size;
  std::vector<std::vector<int>> reconfig_flag; // Deeksha: Added reconfig flag for each layer and region
};


class RegionalTopoManager : public EventSource {
  // regional topology reconfiguration
public:
  enum TopoStatus {
    TOPO_INIT,
    TOPO_LIVE,
    TOPO_RECONF,
    TOPO_WAIT_DRAIN
  };
  RegionalTopoManager(Vermilion* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList & eventlist, int region_id, bool periodic_reconf = false, simtime_picosec periodic_interval = 0, VermilionConnCycleAlgorithm conn_cycle_algorithm = CONN_CYCLE_VERMILION);
  virtual void doNextEvent();
  void start_reconf();
  void finish_reconf();
  void do_reconf();
  void _do_reconf();
  // for regional reconf
  void update_regional_queue_bandwidth();
  void set_regional_queues_pause_recved();
  void set_regional_tcp_pause();
  void update_regional_route();
  void resume_regional_tcp_flows();
  void normalize_tm(Matrix2D<double> & normal_tm);    //Deeksha: Issue with using this
  bool is_in_region(TcpSrc* tcpsrc) const;
  std::vector<std::vector<int>> regional_topo_reconfig();
  std::vector<std::vector<int>> rotor_reconfig();
  void prepare_conn_cycle();
  void compute_schedule_vermilion();
  void compute_schedule_birkhoff();
  void rotor_conn_cycle();
  bool use_rotor_paths() const;
  bool has_active_alltoall_in_region(RegionalTopoManager* region_manager, int region_id);
  int nnodes;
  int region_size;
  int region_num;
  int alpha;
  int start_node;
  int end_node;
  Vermilion* topo;
  int current_layer_id=-1;
  int current_micro_batch_id=-1;

  // vamsi: to togglge the topology direction between dispatch and combine all to all
  int reverse = 0;

  int current_layer_id_2=-1;
  int current_micro_batch_id_2=-1;

  int region_id;
  simtime_picosec reconf_delay;
  simtime_picosec reconfig_end_time;
  bool periodic_reconf;
  simtime_picosec periodic_interval;
  VermilionConnCycleAlgorithm conn_cycle_algorithm;
  bool conn_cycle_ready = false;
  bool conn_cycle_ready_2 = false;
  std::vector<std::vector<std::vector<int>>> conn_cycle;
  std::vector<simtime_picosec> conn_cycle_intervals;
  size_t conn_cycle_idx = 0;
  simtime_picosec active_conn_cycle_interval = 0;
  TopoStatus status;
  EventList & eventlist;
  All2AllTrafficRecorder * demandrecorder;
  uint64_t non_empty_queues;
  //Deeksha:  Number of paused all-to-all flows in this region after calling set_regional_tcp_pause()
  int paused_flow_count = 0;
  // Track whether we've completed the first reconfiguration for this region
  bool first_reconf_done = false;
  std::vector<std::vector<int>> reconfig_conn_matrix;  // Store the conn matrix computed in start_reconf

  bool activeAlltoAll = false;
};

class VermilionTopoManager {
public:
  VermilionTopoManager(Vermilion* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList & eventlist, bool periodic_reconf = false, simtime_picosec periodic_interval = 0, VermilionConnCycleAlgorithm conn_cycle_algorithm = CONN_CYCLE_VERMILION);
  vector<RegionalTopoManager*> regional_topo_managers;
  int nnodes;
  int region_size;
  int region_num;
  int alpha;
  Vermilion* topo;
  simtime_picosec reconf_delay;
  bool periodic_reconf;
  simtime_picosec periodic_interval;
  VermilionConnCycleAlgorithm conn_cycle_algorithm;
  All2AllTrafficRecorder * demandrecorder;
  void transpose(int regionId){
    regional_topo_managers[regionId]->reverse = (regional_topo_managers[regionId]->reverse + 1)%2;
    regional_topo_managers[regionId]->conn_cycle_ready = 0;
  }
};

#endif // VERMILION_TOPO_MANAGER_H
