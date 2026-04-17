// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "mixnet_topomanager.h"
#include <vector>
#include "string.h"
#include <sstream>
#include <fstream>
#include <strstream>
#include <iostream>
// #include "taskgraph.pb.h"

// #include "main.h"
// #include "queue.h"
// #include "switch.h"
// #include "compositequeue.h"
// #include "prioqueue.h"
// #include "queue_lossless.h"
// #include "queue_lossless_input.h"
// #include "queue_lossless_output.h"
// #include "ecnqueue.h"
// #include "taskgraph_generated.h"
// #include <queue>
// #include <utility>
static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<double> unif(0, 1);
// FIXME: make sure all the src and dst are gpu idx, we need to do the conversion inside the node
#define EDGE(a, b, n) ((a) > (b) ? ((a) * (n) + (b)) : ((b) * (n) + (a)))
#define deg 16

extern uint32_t RTT;
extern uint32_t SPEED;
extern ofstream fct_util_out;


All2AllTrafficRecorder::All2AllTrafficRecorder(int layer_num, int region_num, int region_size, TcpRtxTimerScanner *rtx_scanner)
    : rtx_scanner(rtx_scanner), region_size(region_size), region_num(region_num), layer_num(layer_num)
{
  reconfig_flag.resize(layer_num);

  for (int i = 0; i < layer_num; i++)
  {
    reconfig_flag[i].resize(region_num);
    for (int j = 0; j < region_num; j++)
    {
      reconfig_flag[i][j] = 0;
    }
  }
  // init the traffic matrix
  traffic_matrix.resize(layer_num);
  
  for (int i = 0; i < layer_num; i++)
  {
    traffic_matrix[i].resize(region_num);
    for (int j = 0; j < region_num; j++)
    {
      // Directly construct the matrix
      traffic_matrix[i][j] = new Matrix2D<double>(region_size, region_size);
      // Minimal matrix to carry the base flag for reconfig tracking
    }
  }
}

// Deeksha: The following function doesn't update the traffic matrix, per layer per region it stores the matrix exactly once.
void All2AllTrafficRecorder::append_traffic_matrix(int layer_id, int region_id, Matrix2D<double> &tm)
{
  // TODO: check the layerid is from 0 or from 1
  assert(layer_id < layer_num);
  assert(region_id < region_num);
  if (!tm.is_symmetric())
  {
    tm.transform2symmetric(); // Deeksha: Normalise the traffic matrix instead of symmettic
  }
  if (traffic_matrix[layer_id][region_id]->get_base() == 0)
  { // Deeksha: base is just to check if the matrix is initialized 0 means not initialized
    traffic_matrix[layer_id][region_id]->copy_from(tm);
    traffic_matrix[layer_id][region_id]->update_base(1);
    // cout << "Appended traffic matrix at layer " << layer_id
    //      << " region " << region_id << *traffic_matrix[layer_id][region_id] <<std::endl;
  }
  else
  {
    // check the value of tm is eq to the base of the traffic matrix
    if (!traffic_matrix[layer_id][region_id]->check_eq(tm))
    {
      std::cerr << "Error: Traffic matrix mismatch at layer " << layer_id
                << " region " << region_id << std::endl;
      assert(false && "Traffic matrix mismatch");
    }
  }
}

// Deeksha: Normalizing the traffic matrix per layer per region
void All2AllTrafficRecorder::NormalizeTrafficMatrix(int layer_id, int region_id, Matrix2D<double> &tm)
{
  // TODO: check the layerid is from 0 or from 1
  assert(layer_id < layer_num);
  assert(region_id < region_num);

  // Deeksha: TODO check for small values
  // tm.mul_by(alpha) //To do scaling
  tm.normalize();
  tm.mul_by(deg); // Deeksha: Hardcoded with mul by 16, to do change this
  tm.floor_mat(); // Deeksha: Flooring the values to avoid fractional flows
  // cout << "Normalized traffic matrix at layer " << layer_id
  //      << " region " << region_id << std::endl;

  if (traffic_matrix[layer_id][region_id]->get_base() == 0)
  { // Deeksha: base is just to check if the matrix is initialized 0 means not initialized
    traffic_matrix[layer_id][region_id]->copy_from(tm);
    traffic_matrix[layer_id][region_id]->update_base(1);
  }
  else
  {
    // check the value of tm is eq to the base of the traffic matrix
    if (!traffic_matrix[layer_id][region_id]->check_eq(tm))
    {
      std::cerr << "Error: Traffic matrix mismatch at layer " << layer_id
                << " region " << region_id << std::endl;
      assert(false && "Traffic matrix mismatch");
    }
  }
  reconfig_flag[layer_id][region_id] = 0; // Deeksha: Resetting the reconfig flag after normalization
}

// Deeksha: Creating Vermilion graph from the traffic matrix

void All2AllTrafficRecorder::VermilionGraph(int layer_id, int region_id, Matrix2D<double> &tm)
{
  int N = region_size; // region_size; Deeksha: Hardcoded for now, to do change this to region_size
  std::vector<int> out_Left(N);
  std::vector<int> in_Left(N);

  // out_Left: deg - row sums
  for (int row = 0; row < N; ++row)
  {
    long long row_sum = 0;
    for (int col = 0; col < N; ++col)
    {
      row_sum += tm.get_elem(row, col);
    }
    out_Left[row] = static_cast<int>(deg - row_sum);
  }

  // in_Left: deg - column sums
  for (int col = 0; col < N; ++col)
  {
    long long col_sum = 0;
    for (int row = 0; row < N; ++row)
    {
      col_sum += tm.get_elem(row, col);
    }
    in_Left[col] = static_cast<int>(deg - col_sum);
  }
}

bool All2AllTrafficRecorder::check_traffic_matrix(int layer_id, int region_id)
{
  // TODO: check the layerid is from 0 or from 1
  assert(layer_id < layer_num);
  assert(region_id < region_num);

  if (traffic_matrix[layer_id][region_id]->get_base() == 0)
  {
    return false;
  }
  else
  {
    return true;
  }
}
// Deeksha: For debugging purpose
void All2AllTrafficRecorder::dump_all_traffic_matrices(int layer_id, int region_id, Matrix2D<double> &tm) const
{
  for (int l = 0; l < layer_num; l++)
  {
    for (int r = 0; r < region_num; r++)
    {
      if (traffic_matrix[l][r]->get_base() == 0)
      {

        continue; // skip uninitialized matrices
      }
      // traffic_matrix[layer_id][region_id]->copy_from(tm);  //traffic matric is already copied in append function
      std::cout << "Traffic matrix for layer " << l << " region " << r << ":" << std::endl;
      traffic_matrix[l][r]->print_mat();
    }
  }
}

RegionalTopoManager::RegionalTopoManager(Mixnet *topo, All2AllTrafficRecorder *demandrecorder, simtime_picosec refonc_delay, EventList &eventlist, int region_id, bool periodic_reconf, simtime_picosec periodic_interval)
  : EventSource(eventlist, "MixnetRegionalTopoManager"), topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay), eventlist(eventlist), region_id(region_id), periodic_reconf(periodic_reconf), periodic_interval(periodic_interval), paused_flow_count(0), first_reconf_done(false)
{

  nnodes = topo->no_of_nodes();
  region_size = topo->region_size;
  region_num = topo->region_num;
  alpha = topo->alpha;
  status = TopoStatus::TOPO_LIVE;

  start_node = region_id * region_size;
  end_node = start_node + region_size;

  // set the regional topo manager for the queues and pipes
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      // if (i == j)
      //   continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      eq->set_regional_topo_manager(this);
      // std::cerr << "setting queue " << eq << std::endl;
      Pipe *p = topo->pipes[i][j];
      p->set_regional_topo_manager(this);
    }
  }
}

void RegionalTopoManager::doNextEvent()
{
  // TODO: trigger this in ffapp
  std::cerr << "RegionalTopoManager::doNextEvent() status=" << static_cast<int>(status) << " time = " << eventlist.now() <<std::endl;
  if (status == TopoStatus::TOPO_LIVE)
  {
    

    if (periodic_reconf && !conn_cycle_ready)
    {
      prepare_conn_cycle(); // Deeksha: Call this after matrix is ready before this TASK call for doNextEvent
    }
    start_reconf();
    
  }
  else if(status == TopoStatus::TOPO_WAIT_DRAIN) //Deeksha: Added this new state 
  {
    non_empty_queues = 0;
    set_regional_queues_pause_recved(); // recompute drain state without injecting new traffic
      if (non_empty_queues == 0 )
    { 
      _do_reconf();
      status = TopoStatus::TOPO_RECONF;
    }
   
  else{
      eventlist.sourceIsPendingRel(*this, timeFromNs(100)); // Deeksha: Check every 1 us if the queues are drained, to trigger the reconfiguration as soon as possible
      std::cout<<"Stuck here for region "<< region_id << " non_empty_queues: " << non_empty_queues << std::endl;
      status = TopoStatus::TOPO_WAIT_DRAIN;
  }
  }
  else
  {
    finish_reconf();
    status = TopoStatus::TOPO_LIVE;
    reconfig_end_time = 0;
    std::cout<< " Topology reconfiguration finished for region " << region_id << " at time " << eventlist.now() << std::endl;
  }
}

void RegionalTopoManager::start_reconf()
{
  // Compute the conn matrix based on traffic matrix at the beginning
  non_empty_queues = 0;
  set_regional_tcp_pause();           // to be done  //Deeksha tcp flow s paused
  set_regional_queues_pause_recved(); // done   //Deeksha: made changes in this function
  if (non_empty_queues == 0)
  {
     _do_reconf();
    status = TopoStatus::TOPO_RECONF;
  }
   
  else{
    eventlist.sourceIsPendingRel(*this, timeFromNs(100)); // Deeksha: Check every 1 us if the queues are drained, to trigger the reconfiguration as soon as possible
    std::cout<<"Stuck here for region "<< region_id << " non_empty_queues: " << non_empty_queues << std::endl;
    status = TopoStatus::TOPO_WAIT_DRAIN;
  }
  //Deeksha: To test without waiting for drain  -> failed with an assertion error in finish_reconf because the queues are not empty and we are trying to set them to ready state  if we skip this assert another next hop assert issue raises
  // _do_reconf();
  //  status = TopoStatus::TOPO_RECONF;
}

void RegionalTopoManager::do_reconf()
{
  non_empty_queues = 0;
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      if (eq->queuesize() > 0)
      {
        // eq->_state_send = LosslessQueue::PAUSE_RECEIVED;
        non_empty_queues += eq->_enqueued.size();
        // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      }
      else
      {
        // eq->_state_send = LosslessQueue::PAUSED;
      }
      Pipe *p = topo->pipes[i][j];
      if (!p->_inflight.empty())
        non_empty_queues += p->_inflight.size();
    }
  }
  // std::cerr << "non_empty_queues: " << non_empty_queues << std::endl;
  if (non_empty_queues == 0)
  {
    _do_reconf();
  }
}
void RegionalTopoManager::_do_reconf()
{
  // if (periodic_reconf && conn_cycle_ready && !conn_cycle.empty())
  // {
  //   reconfig_conn_matrix = conn_cycle[conn_cycle_idx];         // Updating the regional conn matrix from the conn cycle
  //   conn_cycle_idx = (conn_cycle_idx + 1) % conn_cycle.size(); // Deeksha: Rotate to next conn matrix
  // }
  // else
  // {
  //   reconfig_conn_matrix.clear();     
  // }
  update_regional_queue_bandwidth();
  // refresh precomputed routes to match the new conn matrix
  //topo->renew_routes_ocs(start_node, end_node);
  update_regional_route();
  status = TopoStatus::TOPO_RECONF;
  eventlist.sourceIsPendingRel(*this, reconf_delay);
  reconfig_end_time = eventlist.now() + reconf_delay + 1;
  std::cout << "Region " << region_id <<" layer id " << current_layer_id << " reconfiguration will finish at " << reconfig_end_time << std::endl;
}

void RegionalTopoManager::finish_reconf()
{
  // resume_lively_queues();
  // pause_no_bw_queues();
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      // std::cerr << "queue " << i << ", " << j << " br " << eq->_bitrate << " ps per byte " << eq->_ps_per_byte << " size " << eq->_enqueued.size() << std::endl;
      assert(eq->_enqueued.empty());      //Deeksha: Commented for testing 
      // if (eq->_bitrate > 0)
      {
        eq->_state_send = LosslessQueue::READY;
        // if (!eq->_enqueued.empty())
        // {
        //   eq->beginService();
        //   // std::cerr << "queue " << i << ", " << j << " br " << eq->_bitrate << " ps per byte " << eq->_ps_per_byte << " size " << eq->_enqueued.size() << std::endl;
        //   // std::cerr << "\t starting... " << std::endl;
        // }
      }
    }
  }

  // Deeksha: Triggering after the reconfig within the do next event itself
  
  resume_regional_tcp_flows();
  // Only skip scheduling when there are no paused flows after the first reconfiguration.
  // For the very first reconfiguration we still want to schedule the periodic reconf regardless
  // of paused_flow_count so that the periodic policy takes effect.
  if (periodic_reconf && periodic_interval > 0 && (paused_flow_count > 0 || !first_reconf_done))
  {
    eventlist.sourceIsPending(*this, eventlist.now() + periodic_interval);
    std::cout<< "periodic reconf scheduled at "<< region_id << current_layer_id << " at time " << eventlist.now() + periodic_interval << std::endl;
  }
  else
  {
    std::cout<< "No periodic reconf scheduled at "<< region_id << current_layer_id << " at time " << eventlist.now() << " paused_flow_count: " << paused_flow_count << std::endl;
  }

  // Mark that we've completed at least one reconfiguration for this region
  first_reconf_done = true;
}
//Deelksha: Set the queues and pipes in the region to PAUSE_RECEIVED state
void RegionalTopoManager::set_regional_queues_pause_recved()
{
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      if(eq->queuesize() > 0)
      {
        std::cout << "Queue" << eq->_enqueued.size()
                  << " state=" << eq->_state_send << std::endl;
        // Allow existing backlog to drain; flows are paused so no new packets enter
        // Treat both PAUSED and PAUSE_RECEIVED as needing an explicit resume. // vamsi
        if (eq->_state_send == LosslessQueue::PAUSED ||
            eq->_state_send == LosslessQueue::PAUSE_RECEIVED) {
          eq->_state_send = LosslessQueue::READY;
          if (!eq->_enqueued.empty()) {
            eq->beginService();
          }
        }

        non_empty_queues += eq->_enqueued.size();
        // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      }
      
      Pipe *p = topo->pipes[i][j];
      if (!p->_inflight.empty()){
        // while (!p->_inflight.empty())
        // {
        //   p->_inflight.pop_front();
        // }

        //Do nothing, just counting the inflight packets
        std::cerr << "Draining pipe " << i << ", " << j << " size " << p->_inflight.size() <<std::endl;
        std::cout<< "*Check*" << std::endl;

        // vamsi: only account inflight packets; don't change queue state here
        non_empty_queues += p->_inflight.size();
        //std::cout<<"Inflight packet types in pipe p->type "<< p->type << std::endl;
        std::cout<<"Pipe "<<i<<","<<j<<" size "<<p->_inflight.size()<< std::endl;
      }
        
    }
  }
  // std
}

void RegionalTopoManager::set_regional_tcp_pause()
{
  /*
    refer to DynFlatScheduler::set_all_tcp_pause()
    1. iter the running tcp flows
    2. if the tcp flow is in this region, set the tcp flow to pause
  */
 //std::cout << "Setting regional tcp pause for region " << region_id << std::endl;
  paused_flow_count = 0;
  list<TcpSrc *>::iterator i = demandrecorder->rtx_scanner->_tcps.begin();
  while (i != demandrecorder->rtx_scanner->_tcps.end())
  {
    TcpSrc *tcpsrc = *i;
    if (is_in_region(tcpsrc)&& (!tcpsrc->is_elec))
    {
      assert(tcpsrc->is_all2all);
      // Only pause within this region to avoid tripping the global pause flag
      // that would block unrelated flows.
      tcpsrc->pause_flow_in_region(); // vamsi: use per-flow regional pause instead of global pause
      paused_flow_count++;
      
      
      std::cout << "Pausing flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " at index " << conn_cycle_idx << std::endl;
      if(tcpsrc->is_all2all && tcpsrc->is_elec)
      {
        std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is electric." << std::endl;
      }
      else if(tcpsrc->is_all2all && !tcpsrc->is_elec)
      {
        std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is optical." << std::endl;
      }
      
    }
    // if (tcpsrc->is_all2all && is_in_region(tcpsrc))
    // {
    //   std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is in region " << region_id << "." << std::endl;
    // }
    // if(tcpsrc->is_all2all && tcpsrc->is_elec)
    // {
    //   std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is electric." << std::endl;
    // }
    // else if(tcpsrc->is_all2all && !tcpsrc->is_elec)
    // {
    //   std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is optical." << std::endl;
    // }
    i++;
  }
  std::cout << "For region " << region_id << " Total flows paused: " << paused_flow_count << std::endl;
}


void RegionalTopoManager::update_regional_route()
{
  /*
    refer to DynFlatScheduler::update_all_route()
    iterate the running flows:
      1. allreduce / p2p flow at elec fabric: no need to update
      2. all2all flow:
        - case1: flow is in this region: update the route
        - case2: flow is not in this region: no need to update
  */
  list<TcpSrc *>::iterator i = demandrecorder->rtx_scanner->_tcps.begin();
  while (i != demandrecorder->rtx_scanner->_tcps.end())
  {
    TcpSrc *tcpsrc = *i;
    if (is_in_region(tcpsrc))
    {
      // update path of running flow
      int src = tcpsrc->_flow_src / 8;
      int dst = tcpsrc->_flow_dst / 8;
      if (topo->conn[src][dst] > 0)
      {
        // this flow is re-assigned to ocs fabric
        tcpsrc->is_elec = false;
      }
      else
      {
        // this flow is re-assigned to elec fabric
        tcpsrc->is_elec = false; // Deeksha: Trying to pause all flows which do not have direct connect for now, to avoid the complexity of route update for elec flows, to do: update route for elec flows as well
      }
      assert(tcpsrc->is_all2all);
      Route *routeout, *routein;
      int choice;
      vector<const Route *> *srcpaths = topo->get_paths(tcpsrc->_flow_src, tcpsrc->_flow_dst);
      // choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
      choice = 0; // Deeksha: Choosing the first path
      routeout = new Route(*(srcpaths->at(choice)));
      routeout->push_back(tcpsrc->_sink);
      vector<const Route *> *dstpaths = topo->get_paths(tcpsrc->_flow_dst, tcpsrc->_flow_src);
      // choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
      choice = 0; // Deeksha: Choosing the first path
      routein = new Route(*(dstpaths->at(choice)));
      routein->push_back(tcpsrc);
      tcpsrc->update_route(routeout, routein);

      delete srcpaths;
      delete dstpaths;
    }
    i++;
  }
}
void RegionalTopoManager::resume_regional_tcp_flows()
{
  /*
    refer to DynFlatScheduler::resume_tcp_flows()
  */
 int resumed_flows = 0;
  list<TcpSrc *>::iterator i = demandrecorder->rtx_scanner->_tcps.begin();
  while (i != demandrecorder->rtx_scanner->_tcps.end())
  {
    TcpSrc *tcpsrc = *i;
    if (is_in_region(tcpsrc))
    {
      assert(tcpsrc->is_all2all);
      int src_node = tcpsrc->_flow_src / NUM_GPU_PER_NODE;
      int dst_node = tcpsrc->_flow_dst / NUM_GPU_PER_NODE;
      // Only resume flows that have an active OCS connection in the current conn matrix.
      if (topo->conn[src_node][dst_node] > 0)
      {
        // vamsi: clear regional pause flag and resume the flow
        tcpsrc->resume_flow_in_region();
        tcpsrc->resume_flow();
        std::cout << "Resuming flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " at index " << conn_cycle_idx << std::endl;
        //paused_flow_count--;
        resumed_flows++;
      }
      else
      {
        std::cout << "Keeping flow paused (no OCS conn) from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << std::endl;
      }
    }
    i++;
  }
  std::cout << "For region " << region_id << " Total flows resumed: " << resumed_flows << " out of " << paused_flow_count <<std::endl;
}

void RegionalTopoManager::update_regional_queue_bandwidth()
{
  /*
    refer to DynFlatScheduler::update_all_queue_bandwidth()
    1. use the conn matrix computed in start_reconf()
    2. update the queue bandwidth based on conn matrix
  */
  // Use the pre-computed conn matrix from start_reconf()

    reconfig_conn_matrix = regional_topo_reconfig();
  
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      if (i == j) {
        continue;
      }
      else {
        topo->queues[i+start_node][j+start_node]->_bitrate = reconfig_conn_matrix[i][j] * speedFromMbps((uint64_t)SPEED);
        topo->queues[i+start_node][j+start_node]->_ps_per_byte = (simtime_picosec)((pow(10.0, 12.0) * 8) / topo->queues[i+start_node][j+start_node]->_bitrate);
      }
    }
  }
 //Printing the updated conn matrix for debugging
  std::cout << "Updated conn matrix for region " << region_id << " at layer " << current_layer_id << ":" << std::endl;
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      std::cout << reconfig_conn_matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
  // direct connect no need to update the path
  // topo->renew_routes_ocs(start_node, end_node); // update regional routes
}

bool RegionalTopoManager::is_in_region(TcpSrc *tcpsrc) const{
  // First check if the flow is finished
  if (tcpsrc->_finished)
  {
    return false;
  }
  // Then check if it's from all2all task
  // if (!tcpsrc->is_all2all) {
  //   assert(tcpsrc->is_elec);
  //   return false;
  // }
  if (tcpsrc->is_elec)
  {
    // dont pause elec flows
    return false;
  }
  // Finally check if both source and destination are in this region
  return tcpsrc->_flow_src >= 8 * start_node && tcpsrc->_flow_src < 8 * end_node &&
         tcpsrc->_flow_dst >= 8 * start_node && tcpsrc->_flow_dst < 8 * end_node;
}


// Deeksha: Update conn cycle as per vermilion adjacency matrix generation
void RegionalTopoManager::prepare_conn_cycle()
{ 
  if (!demandrecorder->check_traffic_matrix(current_layer_id, region_id))
  {
    return;
  }

  conn_cycle.clear();
  std::vector<std::vector<std::vector<int>>> conn(6,
    std::vector<std::vector<int>>(region_size,
        std::vector<int>(region_size, 0)));

  // // conn = {
  // //   {// conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    },
  // //   {  // conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    },
  // //    {// conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    },
  // //   {  // conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    },
     
  // //   { // conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    },
  // //   { // conn[5]
  // //    {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1},
  // //      {1,1,1,1,1,1,1,1},
  // //     {1,1,1,1,1,1,1,1}
  // //    } 
  // //   };

  // conn = {
  //   { // conn[0]
  //     {1,1,2,1,1,0,1,0},
  //     {1,1,0,2,0,1,2,0},
  //     {2,0,1,1,0,1,0,1},
  //     {1,2,1,1,0,1,0,0},
  //     {1,0,0,0,2,0,1,1},
  //     {0,1,1,1,0,1,1,2},
  //     {1,2,0,0,1,1,1,1},
  //     {0,0,1,0,1,2,1,1}
  //   },
  //   { // conn[1]
  //     {1,0,1,1,0,1,1,1},
  //     {0,2,0,1,0,1,1,1},
  //     {1,0,1,1,1,1,1,1},
  //     {1,1,1,1,1,2,0,0},
  //     {0,0,1,1,1,1,1,1},
  //     {1,1,1,2,1,1,0,0},
  //     {1,1,1,0,1,0,0,2},
  //     {1,1,1,0,1,0,2,0}
  //   },
  //   { // conn[2]
  //     {0,1,1,1,1,0,1,1},
  //     {1,0,0,1,0,1,1,1},
  //     {1,0,1,1,1,2,0,1},
  //     {1,1,1,0,1,1,1,0},
  //     {1,0,1,1,0,1,1,1},
  //     {0,1,2,1,1,0,1,0},
  //     {1,1,0,1,1,1,0,1},
  //     {1,1,1,0,1,0,1,1}
  //   },
  //   { // conn[3]
  //     {0,1,0,1,1,1,1,1},
  //     {1,0,1,1,1,1,1,0},
  //     {0,1,0,1,2,0,1,0},
  //     {1,1,1,0,0,0,1,1},
  //     {1,1,2,0,1,0,0,1},
  //     {1,1,0,0,0,0,1,1},
  //     {1,1,1,1,0,1,0,0},
  //     {1,0,0,1,1,1,0,1}
  //   },
  //   { // conn[4]
  //     {0,1,0,1,1,1,0,2},
  //     {1,0,1,0,1,1,1,1},
  //     {0,1,0,0,2,0,1,1},
  //     {1,0,0,0,1,1,2,1},
  //     {1,1,2,1,0,0,0,1},
  //     {1,1,0,1,0,0,1,1},
  //     {0,1,1,2,0,1,0,0},
  //     {2,1,1,1,1,1,0,0}
  //   },
  //   { // conn[5]
  //     {0,1,1,1,0,1,0,1},
  //     {1,0,1,0,1,1,2,1},
  //     {1,1,0,1,0,1,1,1},
  //     {1,0,1,0,2,1,0,1},
  //     {0,1,0,2,0,0,2,1},
  //     {1,1,1,1,0,0,1,1},
  //     {0,2,1,0,2,1,0,0},
  //     {1,1,1,1,1,1,0,0}
  //   }
  // };

  // conn = {
  //   { // conn[0]
  //     {1,0,0,1,1,0,1,0},
  //     {0,1,1,0,0,1,0,1},
  //     {1,1,1,0,0,1,0,0},
  //     {0,1,1,1,0,0,0,1},
  //     {1,0,0,0,2,0,1,0},
  //     {0,1,1,0,0,1,1,0},
  //     {0,0,0,2,0,1,1,0},
  //     {1,0,0,0,1,0,0,2}
  //   },
 
  //   { // conn[2]
  //     {1,0,0,1,1,0,1,0},
  //     {0,1,1,0,1,1,0,0},
  //     {0,0,1,1,0,0,0,2},
  //     {1,0,0,0,1,1,1,0},
  //     {0,1,0,1,0,1,1,0},
  //     {1,0,0,0,1,1,0,1},
  //     {1,1,1,0,0,0,0,1},
  //     {0,1,1,1,0,0,1,0}
  //   },
    
  //   { // conn[4]
  //     {0,1,0,1,1,0,1,0},
  //     {1,0,1,0,1,0,1,0},
  //     {0,0,1,1,0,0,0,2},
  //     {1,0,0,0,1,1,1,0},
  //     {0,1,0,1,0,1,0,1},
  //     {1,0,1,0,1,1,0,0},
  //     {1,1,1,0,0,0,0,1},
  //     {0,1,0,1,0,1,1,0}
  //   },
    
  //   { // conn[6]
  //     {0,1,0,0,1,1,0,1},
  //     {1,0,0,1,1,0,1,0},
  //     {0,1,1,1,0,1,0,0},
  //     {1,0,1,0,1,0,0,1},
  //     {0,1,1,1,0,0,1,0},
  //     {1,0,0,0,0,0,2,1},
  //     {0,0,1,0,1,2,0,0},
  //     {1,1,0,1,0,0,0,1}
  //   },
    
  //   { // conn[8]
  //     {0,1,0,0,1,1,1,0},
  //     {1,0,0,1,1,0,0,1},
  //     {0,1,1,0,0,1,1,0},
  //     {0,1,1,0,0,1,1,0},
  //     {1,0,1,0,1,0,0,1},
  //     {0,0,1,2,0,0,0,1},
  //     {2,0,0,1,0,0,0,1},
  //     {0,1,0,0,1,1,1,0}
  //   },
    
  //   { // conn[10]
  //     {0,0,1,0,1,1,0,1},
  //     {0,1,0,1,0,0,1,1},
  //     {1,0,0,0,1,0,2,0},
  //     {1,0,1,0,0,2,0,0},
  //     {1,1,1,0,1,0,0,0},
  //     {0,1,0,2,0,0,0,1},
  //     {1,1,0,1,0,0,0,1},
  //     {0,0,1,0,1,1,1,0}
  //   },
    
  // };
//Deeksha: This is for reverse
  conn = {
    { // conn[0]
      {1,1,0,1,0,1,0,0},
      {0,1,0,0,1,0,1,1},
      {1,0,1,0,2,0,0,0},
      {1,0,1,1,0,1,0,0},
      {0,0,1,1,1,0,0,1},
      {0,0,1,0,0,1,1,1},
      {1,1,0,1,0,0,1,0},
      {0,1,0,0,0,1,1,1}
    },
    { // conn[1]
      {1,0,1,0,0,1,1,0},
      {0,1,0,0,1,1,0,1},
      {0,1,1,0,0,0,0,2},
      {1,1,0,1,1,0,0,0},
      {0,1,0,1,1,0,1,0},
      {0,0,2,0,0,1,1,0},
      {1,0,0,1,1,0,1,0},
      {1,0,0,1,0,1,0,1}
    },
    { // conn[2]
      {1,0,1,1,0,0,1,0},
      {0,1,0,1,0,1,0,1},
      {0,0,1,0,1,1,1,0},
      {0,1,1,0,1,0,0,1},
      {1,0,0,1,0,1,0,1},
      {1,1,0,1,0,0,1,0},
      {1,0,0,0,2,1,0,0},
      {0,1,1,0,0,0,1,1}
    },
    { // conn[3]
      {0,0,0,1,1,0,1,1},
      {1,1,0,1,0,1,0,0},
      {0,0,0,1,1,1,0,1},
      {1,0,1,0,0,0,2,0},
      {0,1,1,0,0,1,0,1},
      {1,0,1,0,1,0,0,1},
      {0,1,1,1,0,0,1,0},
      {1,1,0,0,1,1,0,0}
    },
    { // conn[4]
      {1,0,0,1,1,0,0,1},
      {0,0,2,1,0,1,0,0},
      {1,1,0,0,0,1,1,0},
      {1,0,0,0,1,0,0,2},
      {0,1,0,1,0,0,2,0},
      {0,1,1,0,1,0,1,0},
      {0,1,1,1,0,0,0,1},
      {1,0,0,0,1,2,0,0}
    },
    { // conn[5]
      {0,1,0,1,1,0,0,1},
      {1,0,1,0,0,1,1,0},
      {2,0,0,1,0,1,0,0},
      {0,0,2,0,0,0,1,1},
      {1,0,0,0,0,1,1,1},
      {0,2,0,0,2,0,0,0},
      {0,1,1,1,0,0,0,1},
      {0,0,0,1,1,1,1,0}
    }
  };
  // conn = {
  //   { // conn[0]
  //     {1,0,1,0,1,0,1,0},
  //     {0,1,1,0,1,0,1,0},
  //     {1,1,1,0,0,1,0,0},
  //     {0,1,0,1,1,0,0,1},
  //     {0,1,0,0,1,1,1,0},
  //     {1,0,0,1,0,1,0,1},
  //     {0,0,0,2,0,0,1,1},
  //     {1,0,1,0,0,1,0,1}
  //   },
  //   { // conn[1]
  //     {1,0,1,0,0,1,1,0},
  //     {0,1,0,1,0,1,0,1},
  //     {1,0,1,1,0,0,0,1},
  //     {1,1,0,1,0,0,1,0},
  //     {0,1,0,0,1,1,1,0},
  //     {0,0,1,0,1,1,0,1},
  //     {1,0,0,1,1,0,1,0},
  //     {0,1,1,0,1,0,0,1}
  //   },
  //   { // conn[2]
  //     {1,0,0,1,0,1,1,0},
  //     {0,0,1,1,1,0,0,1},
  //     {1,1,1,0,0,0,0,1},
  //     {0,0,0,1,0,1,1,1},
  //     {1,1,0,0,1,1,0,0},
  //     {0,1,0,0,1,1,1,0},
  //     {0,1,1,0,1,0,0,1},
  //     {1,0,1,1,0,0,1,0}
  //   },
  //   { // conn[3]
  //     {0,1,0,1,0,1,0,1},
  //     {2,0,0,1,1,0,0,0},
  //     {1,0,1,0,1,0,1,0},
  //     {0,0,1,0,1,1,0,1},
  //     {1,0,1,1,0,0,0,1},
  //     {0,1,0,0,0,1,2,0},
  //     {0,1,0,1,0,1,0,1},
  //     {0,1,1,0,1,0,1,0}
  //   },
  //   { // conn[4]
  //     {0,1,0,1,0,0,1,1},
  //     {2,0,0,1,0,1,0,0},
  //     {0,0,1,1,2,0,0,0},
  //     {1,0,0,0,1,0,1,1},
  //     {1,1,1,0,0,1,0,0},
  //     {0,1,0,1,0,0,1,1},
  //     {0,0,1,0,0,2,0,1},
  //     {0,1,1,0,1,0,1,0}
  //   },
  //   { // conn[5]
  //     {0,0,1,0,1,0,1,1},
  //     {0,0,1,0,1,1,1,0},
  //     {1,1,0,1,0,1,0,0},
  //     {1,1,1,0,1,0,0,0},
  //     {0,0,0,0,0,1,2,1},
  //     {1,0,1,1,0,0,0,1},
  //     {0,2,0,1,0,0,0,1},
  //     {1,0,0,1,1,1,0,0}
  //   }
  // };

  //std::vector<int> avail_conn(region_size, alpha);
  for (int k = 0; k < conn.size(); k++)  //Deeksha :CHeck this later
  {
    for(int i = 0; i < region_size; i++) {
      for(int j = 0; j < region_size; j++) {
        // Initialize connection matrix
          conn[k][j][i] = conn[k][i][j];       
      }
    }
        
    conn_cycle.push_back(conn[k]);
  }
  
  conn_cycle_ready = !conn_cycle.empty();
  conn_cycle_idx = 0;
}

//Deeksha: Adding the function here to find if there is any active all-to-all flow in the region, to decide whether we can skip the reconfiguration or not
bool has_active_alltoall_in_region(RegionalTopoManager* region_manager, int region_id) {  
    list<TcpSrc *>::iterator i = region_manager->demandrecorder->rtx_scanner->_tcps.begin();  
    while (i != region_manager->demandrecorder->rtx_scanner->_tcps.end()) {  
        TcpSrc* tcpsrc = *i;  
          
        // Check if it's an all-to-all flow and in the specified region  
        if (tcpsrc->is_all2all && region_manager->is_in_region(tcpsrc)) {  
            return true; // Found active all-to-all in this region  
        }  
        i++;  
    }  
    return false; // No active all-to-all found in this region  
}



void RegionalTopoManager::rotor_conn_cycle()
{ 
  if (!demandrecorder->check_traffic_matrix(current_layer_id, region_id))
  {
    return;
  }

  conn_cycle.clear();
  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  std::vector<int> avail_conn(region_size, alpha);
  for (int k = 1; k < region_size; k++)
  {
    for (int i = 0; i < region_size; i++) {
      for (int j = 0; j < region_size; j++) {
        if (i == j) {
          continue; // No self-connections
        }
        else if(j == (i+k)%region_size){
          conn[i][j] = 1; // Initialize connection matrix
          conn[j][i] = 1;
          break;
        }
      }
    }
    conn_cycle.push_back(conn);
  }
  
  conn_cycle_ready = !conn_cycle.empty();
  conn_cycle_idx = 0;
}


std::vector<std::vector<int>> RegionalTopoManager::regional_topo_reconfig() {

  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  conn = conn_cycle[conn_cycle_idx];         // Updating the regional conn matrix from the conn cycle
  conn_cycle_idx++;
  conn_cycle_idx = (conn_cycle_idx) % conn_cycle.size(); // Deeksha: Rotate to next conn matrix
  
  // update the conn of the topo
  // Debug print for conn matrix
  std::cerr << "Debug - Conn matrix for region " << region_id << " layer " << current_layer_id << ":" << "conn index: " << conn_cycle_idx << "conn matrix size: " << conn_cycle.size() << std::endl;
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      std::cerr << conn[i][j] << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "Available connections: ";
  for (int i = 0; i < region_size; i++) {
    for (int j = i+1; j < region_size; j++) {
      if(conn[i][j] > 0) {
        topo->conn[i+start_node][j+start_node] = conn[i][j];
        topo->conn[j+start_node][i+start_node] = conn[i][j];
      }
      else {
        topo->conn[i+start_node][j+start_node] = 0;
        topo->conn[j+start_node][i+start_node] = 0;
      }
    }
  }
  return conn;
}

std::vector<std::vector<int>> RegionalTopoManager::rotor_reconfig() {

  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  conn = conn_cycle[conn_cycle_idx];         // Updating the regional conn matrix from the conn cycle
  conn_cycle_idx = (conn_cycle_idx + 1) % conn_cycle.size(); // Deeksha: Rotate to next conn matrix
  
  // update the conn of the topo
  // Debug print for conn matrix
  std::cerr << "Debug - Conn matrix for region " << region_id << " layer " << current_layer_id << ":" << std::endl;
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      std::cerr << conn[i][j] << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "Available connections: ";
  for (int i = 0; i < region_size; i++) {
    for (int j = i+1; j < region_size; j++) {
      if(conn[i][j] > 0) {
        topo->conn[i+start_node][j+start_node] = conn[i][j];
        topo->conn[j+start_node][i+start_node] = conn[i][j];
      }
      else {
        topo->conn[i+start_node][j+start_node] = 0;
        topo->conn[j+start_node][i+start_node] = 0;
      }
    }
  }

  return conn;
}

MixnetTopoManager::MixnetTopoManager(Mixnet *topo, All2AllTrafficRecorder *demandrecorder, simtime_picosec refonc_delay, EventList &eventlist, bool periodic_reconf, simtime_picosec periodic_interval)
    : topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay), periodic_reconf(periodic_reconf), periodic_interval(periodic_interval)
{

  region_num = topo->region_num;
  region_size = topo->region_size;
  nnodes = topo->no_of_nodes();
  alpha = topo->alpha;

  // init the regional topo managers
  for (int i = 0; i < region_num; i++)
  {
    regional_topo_managers.push_back(new RegionalTopoManager(topo, demandrecorder, reconf_delay, eventlist, i, periodic_reconf, periodic_interval));
  }
}
