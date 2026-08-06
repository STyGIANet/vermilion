#ifndef VERMILION_H
#define VERMILION_H

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
#include <algorithm>
#include <random>
#include <cstdint>
#include <numeric> 
#include <tcp.h>
#ifndef QT
#define QT
typedef enum {RANDOM, ECN, COMPOSITE, CTRL_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN} queue_type;
#endif
using namespace std;
#define NUM_GPU_PER_NODE 8 // full packet (including header), Bytes
class Vermilion: public Topology {
public:
  vector<Switch*> switchs;
  vector<vector<Pipe*>> pipes;
  vector<vector<Queue*>> queues;
  
  FirstFit* ff;
  Logfile* logfile;
  EventList& eventlist;
  int failed_links;
  int _no_of_nodes;
  queue_type qt;

  Vermilion(int no_of_gpus, mem_b queuesize, Logfile *lg, EventList& ev, FirstFit *fit, queue_type q, simtime_picosec refonc_delay, Topology* elec_topology,int alpha_,int dp_degree_, int tp_degree_, int pp_degree_, int ep_degree_);
  void init_network();

  virtual vector<const Route*>* get_paths(int src, int dest);
  virtual vector<const Route*>* get_two_hop_paths(int src, int dest);
  virtual vector<const Route*>* get_eps_paths(int src, int dest); 
  bool has_two_hop_path(int src_gpu_idx, int dest_gpu_idx) const;

  Pipe * get_pipe(int src, int dst) { return pipes[src][dst]; };
  Queue* alloc_queue(QueueLogger* q, uint64_t speed, mem_b queuesize);

  void count_queue(Queue*);
  void print_path(std::ofstream& paths,int src,const Route* route);
  void random_connect();
  void weighted_connect();//debug only
  void create_one_connect();
  void renew_routes_ocs(int start_node, int end_node);

  vector<int>* get_neighbours(int src) { return NULL;};
  int no_of_nodes() const {return _no_of_nodes;}
  int find_lp_switch(Queue* queue);

  uint32_t _link_speed;
  unordered_map<uint64_t, size_t> _conn_list;// no use...
  unordered_map<uint64_t, vector<vector<size_t>*> > _routes;
  
  // parallel degree
  int dp_degree;
  int tp_degree;
  int pp_degree;
  int ep_degree;
  int region_size;
  int region_num;
  
  //elec switch setting
  Topology* elec_topology;
  int alpha;

private:
  map<Queue*,int> _link_usage;

  int find_destination(Queue* queue);
  void set_params(int no_of_nodes);
  mem_b _queuesize;
};

#endif // VERMILION_H


