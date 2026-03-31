#ifndef TOPOLOGY
#define TOPOLOGY
#include "network.h"
#include <utility>
// interface class - set up functionality for derived topology classes

class Topology {
 public:
  //virtual vector<const Route*>* get_paths(int src,int dest, int choose_ocs)=0;
  virtual vector<const Route*>* get_paths(int src_gpu_idx, int dest_gpu_idx) = 0;
  virtual vector<const Route*>* get_eps_paths(int src_gpu_idx, int dest_gpu_idx) {return NULL;} // only mixnet will call this function
  virtual vector<int>* get_neighbours(int src) = 0;  
  virtual int no_of_nodes() const { abort();};
  virtual std::pair<size_t, size_t> change_src_dest(size_t src,size_t dest){ std::cout << "other topo should not call this function: change_src_dest:ERROR" << std::endl; assert(false);}
  virtual std::pair<size_t, size_t> change_src_dest2(size_t src,size_t dest){ std::cout << "other topo should not call this function: change_src_dest:ERROR" << std::endl; assert(false);}
  std::vector<std::vector<int>> conn; // node connections matrix

  // we no longer use this matrix anymore
  std::vector<std::vector<std::vector<std::pair<int,int>>>> expert2gpu;//allexpert matrix
  std::vector<std::vector<std::vector<std::pair<int,int>>>> gpu2expert;//allexpert matrix
};

#endif
