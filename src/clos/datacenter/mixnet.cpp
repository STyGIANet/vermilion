// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "flat_topology.h"
#include <vector>
#include "string.h"
#include <sstream>
#include <fstream>
#include <strstream>
#include <iostream>
// #include "taskgraph.pb.h"

#include "main.h"
#include "queue.h"
#include "switch.h"
#include "compositequeue.h"
#include "prioqueue.h"
#include "queue_lossless.h"
#include "queue_lossless_input.h"
#include "queue_lossless_output.h"
#include "ecnqueue.h"
#include "mixnet.h"
#include "taskgraph_generated.h"
#include <queue>
#include <utility>
static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<double> unif(0, 1);

#define EDGE(a, b, n) ((a) > (b) ? ((a) * (n) + (b)) : ((b) * (n) + (a)))

extern uint32_t RTT;
extern uint32_t SPEED;
extern ofstream fct_util_out;

string ntoa(double n);
string itoa(uint64_t n);

void check_non_null_ocs(Route *rt)
{
  int fail = 0;
  for (unsigned int i = 1; i < rt->size() - 1; i += 2)
    if (rt->at(i) == NULL)
    {
      fail = 1;
      break;
    }

  if (fail)
  {
    for (unsigned int i = 1; i < rt->size() - 1; i += 2)
      printf("%p ", rt->at(i));

    cout << endl;
    assert(0);
  }
}


void Mixnet::set_params(int no_of_gpus)
{
  _no_of_nodes = no_of_gpus / NUM_GPU_PER_NODE;

  switchs.resize(_no_of_nodes, nullptr);
  pipes.resize(_no_of_nodes, vector<Pipe *>(_no_of_nodes));
  queues.resize(_no_of_nodes, vector<Queue *>(_no_of_nodes));
  region_size = ep_degree * tp_degree / NUM_GPU_PER_NODE;
  region_num = _no_of_nodes / region_size;
}

Mixnet::Mixnet(int no_of_gpus, mem_b queuesize, Logfile *lg, EventList &ev, FirstFit *fit, queue_type q, simtime_picosec delay,Topology* elec_topology_, int alpha_, int dp_degree_, int tp_degree_, int pp_degree_, int ep_degree_):eventlist(ev)
{
  _queuesize = queuesize;
  logfile = lg;
  // eventlist = ev;
  ff = fit;
  qt = q;
  failed_links = 0;
  // reconf_delay = delay;
  elec_topology = elec_topology_; // elec network
  alpha = alpha_;
  
  dp_degree=dp_degree_;
  tp_degree=tp_degree_;
  pp_degree=pp_degree_;
  ep_degree=ep_degree_;

  conn = std::vector<std::vector<int>>(no_of_gpus / NUM_GPU_PER_NODE, std::vector<int>(no_of_gpus / NUM_GPU_PER_NODE, 0));

  set_params(no_of_gpus);

  create_one_connect();//Deeksha: create one-to-one connections
  //random_connect();// generate random conn
  // weighted_connect();// generate weighted conn
  init_network();
}


Queue *Mixnet::alloc_queue(QueueLogger *queueLogger, uint64_t speed, mem_b queuesize)
{
  if (qt == RANDOM)
    return new RandomQueue(speedFromMbps(speed), memFromPkt(SWITCH_BUFFER + RANDOM_BUFFER), eventlist, queueLogger, memFromPkt(RANDOM_BUFFER));
  else if (qt == COMPOSITE)
    return new CompositeQueue(speedFromMbps(speed), queuesize, eventlist, queueLogger);
  else if (qt == CTRL_PRIO)
    return new CtrlPrioQueue(speedFromMbps(speed), queuesize, eventlist, queueLogger);
  else if (qt == ECN)
    return new ECNQueue(speedFromMbps(speed), memFromPkt(queuesize), eventlist, queueLogger, memFromPkt(50));
  else if (qt == LOSSLESS)
    return new LosslessQueue(speedFromMbps(speed), memFromPkt(50), eventlist, queueLogger, NULL);
  else if (qt == LOSSLESS_INPUT)
    return new LosslessOutputQueue(speedFromMbps(speed), memFromPkt(200), eventlist, queueLogger);
  else if (qt == LOSSLESS_INPUT_ECN)
    return new LosslessOutputQueue(speedFromMbps(speed), memFromPkt(10000), eventlist, queueLogger, 1, memFromPkt(16));
  assert(0);
}

void Mixnet::init_network()
{
  QueueLoggerSampling *queueLogger = nullptr;

  for (int j = 0; j < _no_of_nodes; j++)
    for (int k = 0; k < _no_of_nodes; k++)
    {
      queues[j][k] = nullptr;
      pipes[j][k] = nullptr;
    }

  //create switches if we have lossless operation
  if (qt == LOSSLESS)
    for (int j = 0; j < _no_of_nodes; j++)
    {
      switchs[j] = new Switch("Switch_LowerPod_" + ntoa(j));
    }

  for (int j = 0; j < _no_of_nodes; j++)
  {
    for (int k = 0; k < _no_of_nodes; k++)
    {
      if (conn[j][k] >= 0) // for all
      {
        queues[j][k] = alloc_queue(queueLogger, SPEED * conn[j][k], _queuesize);
        cerr << "(" << j << ", " << k << ")" << SPEED * conn[j][k] << endl;
        queues[k][j] = alloc_queue(queueLogger, SPEED * conn[j][k], _queuesize);
        queues[j][k]->setName("L" + ntoa(j) + "->DST" + ntoa(k));
        queues[k][j]->setName("L" + ntoa(k) + "->DST" + ntoa(j));
        // logfile->writeName(*(queues[j][k]));
        // logfile->writeName(*(queues[k][j]));

        pipes[j][k] = new Pipe(timeFromNs(RTT), eventlist);
        pipes[k][j] = new Pipe(timeFromNs(RTT), eventlist);
        pipes[j][k]->setName("Pipe-LS" + ntoa(j) + "->DST" + ntoa(k));
        pipes[k][j]->setName("Pipe-LS" + ntoa(k) + "->DST" + ntoa(j));
        // logfile->writeName(*(pipes[j][k]));
        // logfile->writeName(*(pipes[k][j]));

        if (qt == LOSSLESS)
        {
          switchs[j]->addPort(queues[j][k]);
          ((LosslessQueue *)queues[j][k])->setRemoteEndpoint(queues[k][j]);
          switchs[k]->addPort(queues[k][j]);
          ((LosslessQueue *)queues[k][j])->setRemoteEndpoint(queues[j][k]);
        }
        else if (qt == LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN)
        {
          //no virtual queue needed at server
          new LosslessInputQueue(eventlist, queues[k][j]);
          new LosslessInputQueue(eventlist, queues[j][k]);
        }

        if (ff)
        {
          ff->add_queue(queues[j][k]);
          ff->add_queue(queues[k][j]);
        }
      }
    }
  }

  //init thresholds for lossless operation
  if (qt == LOSSLESS)
    for (int j = 0; j < _queuesize; j++)
    {
      switchs[j]->configureLossless();
    }
}


vector<const Route *> *Mixnet::get_paths(int src_gpu_idx, int dest_gpu_idx) {
  // convert the gpu id into node id
  int src = src_gpu_idx / NUM_GPU_PER_NODE;
  int dest = dest_gpu_idx / NUM_GPU_PER_NODE;
  assert(src/region_size == dest/region_size);
  
  route_t *routeout, *routeback;
  if(conn[src][dest]>0) { //allocate ocs path
    vector<const Route *> *paths = new vector<const Route *>();

    route_t *routeout, *routeback;

    // NOTE: HARD CODED `0` BECAUSE THERE'S ONLY ONE SWITCH
    cerr << "id: " << src * _no_of_nodes + dest << endl;
    assert(_routes.find(src * _no_of_nodes + dest) != _routes.end());

    for (const vector<size_t> *r : _routes[src * _no_of_nodes + dest])
    {
      // forward path
      routeout = new Route();
      //routeout->push_back(pqueue);
      const vector<size_t> &route = *r;

      for (size_t i = 0; i < route.size() - 1; i++)
      {
        assert(queues[route[i]][route[i + 1]] != nullptr);
        assert(queues[route[i]][route[i + 1]]->_bitrate!=0);
        routeout->push_back(queues[route[i]][route[i + 1]]);
        routeout->push_back(pipes[route[i]][route[i + 1]]);
      }

      if (qt == LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN)
        routeout->push_back(queues[src][dest]->getRemoteEndpoint());

      routeback = new Route();

      for (size_t i = route.size() - 1; i > 1; i--)
      {
        assert(queues[route[i]][route[i - 1]] != nullptr);
        routeback->push_back(queues[route[i]][route[i - 1]]);
        routeback->push_back(pipes[route[i]][route[i - 1]]);
      }

      if (qt == LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN)
        routeback->push_back(queues[dest][src]->getRemoteEndpoint());

      routeout->set_reverse(routeback);
      routeback->set_reverse(routeout);

      check_non_null_ocs(routeout);
      paths->push_back(routeout);
    }
    return paths;
  }
  else { // allocate ecs path
    // fall back to electrical fabric when no OCS circuit
    return get_eps_paths(src_gpu_idx, dest_gpu_idx);
  }
}

vector<const Route*>* Mixnet::get_eps_paths(int src_gpu_idx, int dest_gpu_idx) {
  // all2all task and p2p task will call this function
  // convert the gpu id into node id
  int src = src_gpu_idx / NUM_GPU_PER_NODE;
  int dest = dest_gpu_idx / NUM_GPU_PER_NODE;
  return elec_topology->get_paths(src, dest);
}




void Mixnet::count_queue(Queue *queue) {
  if (_link_usage.find(queue) == _link_usage.end()) {
    _link_usage[queue] = 0;
  }
  _link_usage[queue] = _link_usage[queue] + 1;
}

// Find lower pod switch:
int Mixnet::find_lp_switch(Queue *queue)
{
  //first check ns_nlp
  for (int i = 0; i < _no_of_nodes; i++)
    for (int j = 0; j < _no_of_nodes; j++)
      if (queues[i][j] == queue)
        return j;

  //only count nup to nlp
  count_queue(queue);

  return -1;
}

int Mixnet::find_destination(Queue *queue)
{
  //first check nlp_ns
  for (int i = 0; i < _no_of_nodes; i++)
    for (int j = 0; j < _no_of_nodes; j++)
      if (queues[i][j] == queue)
        return j;

  return -1;
}

void Mixnet::print_path(std::ofstream &paths, int src, const Route *route)
{
  paths << "SRC_" << src << " ";

  if (route->size() / 2 == 2)
  {
    paths << "LS_" << find_lp_switch((Queue *)route->at(1)) << " ";
    paths << "DST_" << find_destination((Queue *)route->at(3)) << " ";
  }
  else
  {
    paths << "Wrong hop count " << ntoa(route->size() / 2);
  }

  paths << endl;
}

void Mixnet::weighted_connect() {
  // Hardcoded traffic matrix (8x8)
  std::vector<std::vector<double>> init_weight_data = {
    {9810, 5338, 1231, 3042, 4452, 2004, 841, 6050},
    {9278, 6282, 839, 3722, 3821, 2903, 909, 5014},
    {10684, 5165, 1044, 2634, 4344, 3333, 675, 4889},
    {10897, 5629, 1086, 2114, 2762, 3463, 1059, 5758},
    {8684, 5666, 1046, 3955, 4334, 2952, 944, 5187},
    {8361, 5908, 777, 3043, 3984, 3422, 897, 6376},
    {10208, 5507, 1117, 4148, 3967, 3119, 904, 3798},
    {9781, 5211, 902, 3343, 4289, 2839, 1008, 5395}
  };

  // Initialize weight matrix
  Matrix2D<double> init_weight_matrix(ep_degree, ep_degree);
  for (size_t i = 0; i < ep_degree; ++i) {
    for (size_t j = 0; j < ep_degree; ++j) {
      init_weight_matrix.set_elem(i, j, init_weight_data[i][j]);
    }
  }

  std::cerr << "init_weight_matrix " << std::endl;
  std::cerr << init_weight_matrix << std::endl;

  // Initialize conn matrix to 0
  for (int i = 0; i < _no_of_nodes; i++) {
    for (int j = 0; j < _no_of_nodes; j++) {
      conn[i][j] = 0;
    }
  }

  // Initialize expert traffic matrix for full node
  Matrix2D<double> expert_traffic_matrix(dp_degree * ep_degree, dp_degree * ep_degree);
  
  // Replicate the traffic matrix for each DP group
  for (size_t i = 0; i < dp_degree; ++i) {
    for (size_t row = 0; row < ep_degree; ++row) {
      for (size_t col = 0; col < ep_degree; ++col) {
        expert_traffic_matrix.set_elem(i * ep_degree + row, i * ep_degree + col, 
                                       init_weight_matrix.get_elem(row, col));
      }
    }
  }

  // Symmetrize the matrix: add upper triangle to lower triangle
  for (size_t i = 0; i < dp_degree * ep_degree; ++i) {
    for (size_t j = 0; j < dp_degree * ep_degree; ++j) {
      if (i > j) {
        double new_value = expert_traffic_matrix.get_elem(i, j) + expert_traffic_matrix.get_elem(j, i);
        expert_traffic_matrix.set_elem(i, j, new_value);
      }
    }
  }

  // Zero out upper triangle
  for (size_t i = 0; i < dp_degree * ep_degree; ++i) {
    for (size_t j = 0; j < dp_degree * ep_degree; ++j) {
      if (i < j) {
        expert_traffic_matrix.set_elem(i, j, 0);
      }
    }
  }

  std::cerr << "expert_traffic_matrix " << std::endl;
  std::cerr << expert_traffic_matrix << std::endl;

  // Calculate available connections
  int GPU_per_NODE = NUM_GPU_PER_NODE;
  std::vector<int> avail_conn(_no_of_nodes*GPU_per_NODE / (GPU_per_NODE * pp_degree), alpha);
  
  double num_ocs = 1.0 * alpha * tp_degree / 8;
  int ocs_per_expert = (int)std::ceil(num_ocs);
  std::vector<int> avail_connexp(_no_of_nodes*GPU_per_NODE / (tp_degree * pp_degree), ocs_per_expert);
  
  std::cerr << "ocs_per_expert: " << ocs_per_expert << std::endl;

  // Allocate expert-to-expert connections
  while (std::accumulate(avail_conn.begin(), avail_conn.end(), 0) > 0 && 
         std::accumulate(avail_connexp.begin(), avail_connexp.end(), 0) > 0) {
    
    std::vector<std::pair<size_t, size_t>> loc = expert_traffic_matrix.get_sorted_indices();
    int is_find = 0;
    int from_idx = 0;  // 初始化为0，避免未定义行为
    int to_idx = 0;    // 初始化为0，避免未定义行为

    // Find the highest traffic pair that can be connected
    for (auto it = loc.begin(); it != loc.end(); ++it) {
      from_idx = (*it).first;
      to_idx = (*it).second;
      
      // Skip if same expert or on same machine
      if (from_idx == to_idx || from_idx * tp_degree / 8 == to_idx * tp_degree / 8) {
        continue;
      } else {
        is_find = 1;
        break;
      }
    }

    // 如果没有找到有效的连接，直接退出
    if (is_find == 0 || expert_traffic_matrix.get_elem(from_idx, to_idx) == 0) {
      break;
    }
    printf("from_idx: %d, to_idx: %d, expert_traffic_matrix.get_elem(from_idx, to_idx): %f\n", from_idx, to_idx, expert_traffic_matrix.get_elem(from_idx, to_idx));
    assert(is_find != 0);
    
    int from_node = from_idx * tp_degree / 8;
    int to_node = to_idx * tp_degree / 8;

    // Allocate connection if available
    if (avail_conn[from_node] > 0 && avail_conn[to_node] > 0 && 
        avail_connexp[from_idx] > 0 && avail_connexp[to_idx] > 0) {
      
      conn[from_node][to_node] += 1;
      conn[to_node][from_node] += 1;
      avail_conn[from_node] -= 1;
      avail_conn[to_node] -= 1;
      avail_connexp[from_idx] -= 1;
      avail_connexp[to_idx] -= 1;
      
      double new_value = expert_traffic_matrix.get_elem(from_idx, to_idx) / (conn[from_node][to_node]*100);
      printf("from_node: %d, to_node: %d, new_value: %f\n", from_node, to_node, new_value);
      expert_traffic_matrix.set_elem(from_idx, to_idx, new_value);
      printf("after set_elem - from_idx: %d, to_idx: %d, expert_traffic_matrix.get_elem(from_idx, to_idx): %f\n", from_idx, to_idx, expert_traffic_matrix.get_elem(from_idx, to_idx));
    }

    // Zero out row/col if no more connections available
    if (avail_conn[from_node] == 0 || avail_connexp[from_idx] == 0) {
      expert_traffic_matrix.set_row(from_idx, 0);
      expert_traffic_matrix.set_col(from_idx, 0);
      printf("from_idx: %d, set row and col to 0\n", from_idx);
    }
    if (avail_conn[to_node] == 0 || avail_connexp[to_idx] == 0) {
      expert_traffic_matrix.set_row(to_idx, 0);
      expert_traffic_matrix.set_col(to_idx, 0);
      printf("to_idx: %d, set row and col to 0\n", to_idx);
    }
  }

  // Print connection matrix
  std::cerr << "Debug - Conn matrix (weighted_connect):" << std::endl;
  assert(conn.size() != 0);
  for (size_t i = 0; i < conn.size(); ++i) {
    for (size_t j = 0; j < conn.size(); ++j) {
      std::cerr << conn[i][j] << " ";
    }
    std::cerr << std::endl;
  }

  std::cerr << "\nNon-zero connections:" << std::endl;
  for (size_t i = 0; i < conn.size(); ++i) {
    for (size_t j = 0; j < conn[i].size(); ++j) {
      if (conn[i][j] > 0) {
        std::cerr << "  conn[" << i << "][" << j << "] = " << conn[i][j] << std::endl;
      }
    }
  }

  // Renew routes for OCS connections
  renew_routes_ocs(0, _no_of_nodes);
}

void Mixnet::create_one_connect() {
  for (int r = 0; r < region_num; r++) {
    int region_start = r * region_size;
    int region_end = (r + 1) * region_size;
    
    // For each node in the region
    for (int i = region_start; i < region_end; i++) {
      // Count current connections for this node
      for (int j = region_start; j < region_end; j++) {
        if (i==j)
        { 
          conn[i][j] = 0;
          continue;
        } 
        conn[i][j] = 1;
        conn[j][i] = 1;
       }
      }
  }
  renew_routes_ocs(0, _no_of_nodes);
}


void Mixnet::random_connect() {
  /*
      1. regional random connect (there is no connection between regions)
      2. each node can have at most alpha connections
      3. the matrix is symmetric
  */
  
  // For each region
  for (int r = 0; r < region_num; r++) {
    int region_start = r * region_size;
    int region_end = (r + 1) * region_size;
    
    // For each node in the region
    for (int i = region_start; i < region_end; i++) {
      // Count current connections for this node
      int current_conn = 0;
      for (int j = region_start; j < region_end; j++) {
        if (conn[i][j] == 1) current_conn++;
      }
      
      // Try to add more connections up to alpha
      int max_attempts = 100; // Prevent infinite loop
      int attempts = 0;
      while (current_conn < alpha && attempts < max_attempts) {
        // Randomly select a node in the same region
        int j = region_start + (std::rand() % region_size);
        
        // Skip if it's the same node or already connected
        if (i == j || conn[i][j] == 1) {
          attempts++;
          continue;
        }
        
        // Check if target node already has alpha connections
        int target_conn = 0;
        for (int k = region_start; k < region_end; k++) {
          if (conn[j][k] == 1) target_conn++;
        }
        if (target_conn >= alpha) {
          attempts++;
          continue;
        }
        
        // Add connection (both directions for symmetry)
        conn[i][j] = 1;
        conn[j][i] = 1;
        current_conn++;
        attempts = 0; // Reset attempts counter on successful connection
      }
    }
  }
  renew_routes_ocs(0, _no_of_nodes);
}

void Mixnet::renew_routes_ocs(int start_node, int end_node) {
  for(int i=start_node; i < end_node; i++){
    for(int j=start_node; j < end_node; j++){
      int idx=i*_no_of_nodes+j;
      if(i!=j) {
        vector<size_t> *path_vector = new vector<size_t>();
        path_vector->push_back(i);
        path_vector->push_back(j);
        if (_routes.find(idx) == _routes.end())
        {
          _routes[idx] = vector<vector<size_t> *>();
        }
        else {
          for (vector<size_t> *r : _routes[idx])
          {
            delete r;
          }
          _routes[idx].clear();
        }
        assert(path_vector->size() > 0);
        _routes[idx].push_back(path_vector);
      }
    }
  }
}
