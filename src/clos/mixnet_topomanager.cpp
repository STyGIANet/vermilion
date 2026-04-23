// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "mixnet_topomanager.h"
#include <vector>
#include "string.h"
#include <sstream>
#include <fstream>
#include <strstream>
#include <iostream>
#include <cmath>
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
// static std::random_device rd;
// static std::mt19937 gen = std::mt19937(rd());
// static std::uniform_real_distribution<double> unif(0, 1);
// FIXME: make sure all the src and dst are gpu idx, we need to do the conversion inside the node
#define EDGE(a, b, n) ((a) > (b) ? ((a) * (n) + (b)) : ((b) * (n) + (a)))
#define deg 16

extern uint32_t RTT;
extern uint32_t SPEED;
extern ofstream fct_util_out;

namespace {

const char *conn_cycle_algorithm_name(MixnetConnCycleAlgorithm algorithm) {
  switch (algorithm) {
  case CONN_CYCLE_VERMILION:
    return "vermilion";
  case CONN_CYCLE_BIRKHOFF:
    return "birkhoff";
  case CONN_CYCLE_ROTOR:
    return "rotor";
  default:
    return "unknown";
  }
}

std::vector<const Route *> *get_mixnet_two_hop_paths(
    Mixnet *mixnet,
    int src_gpu_idx,
    int dest_gpu_idx) {
  int src = src_gpu_idx / NUM_GPU_PER_NODE;
  int dest = dest_gpu_idx / NUM_GPU_PER_NODE;
  assert(src / mixnet->region_size == dest / mixnet->region_size);

  std::vector<const Route *> *paths = new std::vector<const Route *>();
  const int region_start = (src / mixnet->region_size) * mixnet->region_size;
  const int region_end = region_start + mixnet->region_size;

  for (int mid = region_start; mid < region_end; ++mid)
  {
    if (mid == src || mid == dest)
    {
      continue;
    }

    Route *routeout = new Route();
    routeout->push_back(mixnet->queues[src][mid]);
    routeout->push_back(mixnet->pipes[src][mid]);
    routeout->push_back(mixnet->queues[mid][dest]);
    routeout->push_back(mixnet->pipes[mid][dest]);
    if (mixnet->qt == LOSSLESS_INPUT || mixnet->qt == LOSSLESS_INPUT_ECN)
      routeout->push_back(mixnet->queues[mid][dest]->getRemoteEndpoint());

    Route *routeback = new Route();
    routeback->push_back(mixnet->queues[dest][mid]);
    routeback->push_back(mixnet->pipes[dest][mid]);
    routeback->push_back(mixnet->queues[mid][src]);
    routeback->push_back(mixnet->pipes[mid][src]);
    if (mixnet->qt == LOSSLESS_INPUT || mixnet->qt == LOSSLESS_INPUT_ECN)
      routeback->push_back(mixnet->queues[mid][src]->getRemoteEndpoint());

    routeout->set_reverse(routeback);
    routeback->set_reverse(routeout);
    paths->push_back(routeout);
  }

  return paths;
}

bool has_mixnet_two_hop_path(
    const Mixnet *mixnet,
    int src_gpu_idx,
    int dest_gpu_idx) {
  const int src = src_gpu_idx / NUM_GPU_PER_NODE;
  const int dest = dest_gpu_idx / NUM_GPU_PER_NODE;
  if (src / mixnet->region_size != dest / mixnet->region_size)
  {
    return false;
  }
  return mixnet->region_size > 2;
}

std::vector<std::vector<double>> load_demand_matrix_for_conn_cycle(
    const Matrix2D<double> &traffic_matrix,
    int n,
    bool reverse) {
  const double eps = 1e-5;
  std::vector<std::vector<double>> demand(n, std::vector<double>(n, 0.0));
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      double value = reverse ? traffic_matrix.get_elem(col, row)
                             : traffic_matrix.get_elem(row, col);
      if (row == col || std::fabs(value) < eps) {
        value = 0.0;
      }
      demand[row][col] = value;
    }
  }
  return demand;
}

std::vector<std::vector<double>> normalize_demand_matrix(
    const std::vector<std::vector<double>> &demand) {
  const int n = static_cast<int>(demand.size());
  double max_row_sum = 0.0;
  double max_col_sum = 0.0;

  for (int row = 0; row < n; ++row) {
    double row_sum = 0.0;
    for (int col = 0; col < n; ++col) {
      row_sum += demand[row][col];
    }
    max_row_sum = std::max(max_row_sum, row_sum);
  }

  for (int col = 0; col < n; ++col) {
    double col_sum = 0.0;
    for (int row = 0; row < n; ++row) {
      col_sum += demand[row][col];
    }
    max_col_sum = std::max(max_col_sum, col_sum);
  }

  const double max_sum = std::max(max_row_sum, max_col_sum);
  std::vector<std::vector<double>> normalized = demand;
  if (max_sum == 0.0) {
    return normalized;
  }

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      normalized[row][col] /= max_sum;
    }
  }
  return normalized;
}

std::vector<std::vector<int>> floor_scaled_demand_matrix(
    const std::vector<std::vector<double>> &normalized_demand,
    int scaled_degree) {
  const int n = static_cast<int>(normalized_demand.size());
  std::vector<std::vector<int>> rounded(n, std::vector<int>(n, 0));
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (row == col) {
        continue;
      }
      rounded[row][col] =
          static_cast<int>(std::floor(normalized_demand[row][col] * scaled_degree));
    }
  }
  return rounded;
}

std::vector<std::vector<int>> directed_configuration_model_counts(
    const std::vector<int> &in_left,
    const std::vector<int> &out_left,
    const std::vector<std::vector<int>> &rounded_matrix) {
  const int n = static_cast<int>(in_left.size());
  std::vector<int> out_stubs;
  std::vector<int> in_remaining = in_left;
  int total_in = 0;
  int total_out = 0;

  for (int node = 0; node < n; ++node) {
    assert(in_left[node] >= 0);
    assert(out_left[node] >= 0);
    total_in += in_left[node];
    total_out += out_left[node];
    for (int cnt = 0; cnt < out_left[node]; ++cnt) {
      out_stubs.push_back(node);
    }
  }

  assert(total_in == total_out);
  // Old randomized version:
  // std::shuffle(out_stubs.begin(), out_stubs.end(), gen);
  // Deterministic version: handle higher-residual sources first, then by node id.
  std::stable_sort(out_stubs.begin(), out_stubs.end(),
                   [&out_left](int lhs, int rhs) {
                     if (out_left[lhs] != out_left[rhs]) {
                       return out_left[lhs] > out_left[rhs];
                     }
                     return lhs < rhs;
                   });

  std::vector<std::vector<int>> counts(n, std::vector<int>(n, 0));
  std::vector<int> assigned_in(out_stubs.size(), -1);
  for (size_t idx = 0; idx < out_stubs.size(); ++idx) {
    const int src = out_stubs[idx];
    int best_remaining = -1;
    int chosen_dst = -1;
    int chosen_demand = std::numeric_limits<int>::max();
    for (int dst = 0; dst < n; ++dst) {
      if (dst == src || in_remaining[dst] <= 0) {
        continue;
      }
      const int candidate_demand = rounded_matrix[src][dst];
      if (in_remaining[dst] > best_remaining ||
          (in_remaining[dst] == best_remaining &&
           (candidate_demand < chosen_demand ||
            (candidate_demand == chosen_demand &&
             (chosen_dst == -1 || dst < chosen_dst))))) {
        best_remaining = in_remaining[dst];
        chosen_demand = candidate_demand;
        chosen_dst = dst;
      }
    }

    if (chosen_dst != -1) {
      // Old randomized tie-break:
      // std::uniform_int_distribution<int> pick_candidate(
      //     0, static_cast<int>(candidates.size()) - 1);
      // const int dst = candidates[pick_candidate(gen)];
      assigned_in[idx] = chosen_dst;
      in_remaining[chosen_dst]--;
      counts[src][chosen_dst]++;
      continue;
    }

    // If only self-capacity remains, repair by swapping one earlier off-diagonal
    // assignment. Prefer disturbing the least-demand edge in the rounded demand
    // matrix; if several candidates tie, use the farthest earlier one.
    int swap_idx = -1;
    int best_swap_demand = std::numeric_limits<int>::max();
    for (int prev = static_cast<int>(idx) - 1; prev >= 0; --prev) {
      const int prev_src = out_stubs[prev];
      const int prev_dst = assigned_in[prev];
      if (prev_src != src && prev_dst != src) {
        const int candidate_demand = rounded_matrix[prev_src][prev_dst];
        if (candidate_demand < best_swap_demand ||
            (candidate_demand == best_swap_demand &&
             (swap_idx == -1 || prev < swap_idx))) {
          best_swap_demand = candidate_demand;
          swap_idx = prev;
        }
      }
    }

    assert(swap_idx != -1);
    assert(in_remaining[src] > 0);

    const int moved_dst = assigned_in[swap_idx];
    assigned_in[swap_idx] = src;
    counts[out_stubs[swap_idx]][moved_dst]--;
    counts[out_stubs[swap_idx]][src]++;

    assigned_in[idx] = moved_dst;
    in_remaining[src]--;
    counts[src][moved_dst]++;
  }

  for (int node = 0; node < n; ++node) {
    assert(in_remaining[node] == 0);
    assert(counts[node][node] == 0);
  }
  return counts;
}

bool find_augmenting_path(
    int u,
    const std::vector<std::vector<int>> &edge_counts,
    std::vector<int> &match_right,
    std::vector<char> &seen) {
  const int n = static_cast<int>(edge_counts.size());
  for (int v = 0; v < n; ++v) {
    if (u == v || edge_counts[u][v] <= 0 || seen[v]) {
      continue;
    }
    seen[v] = 1;
    if (match_right[v] == -1 ||
        find_augmenting_path(match_right[v], edge_counts, match_right, seen)) {
      match_right[v] = u;
      return true;
    }
  }
  return false;
}

bool try_maximum_bipartite_matching(
    const std::vector<std::vector<int>> &edge_counts,
    std::vector<std::pair<int, int>> &matching) {
  const int n = static_cast<int>(edge_counts.size());
  std::vector<int> match_right(n, -1);

  for (int u = 0; u < n; ++u) {
    std::vector<char> seen(n, 0);
    const bool matched = find_augmenting_path(u, edge_counts, match_right, seen);
    if (!matched) {
      matching.clear();
      return false;
    }
  }

  matching.clear();
  matching.reserve(n);
  for (int v = 0; v < n; ++v) {
    if (match_right[v] == -1) {
      matching.clear();
      return false;
    }
    matching.push_back({match_right[v], v});
  }
  return true;
}

std::vector<std::pair<int, int>> maximum_bipartite_matching(
    const std::vector<std::vector<int>> &edge_counts) {
  std::vector<std::pair<int, int>> matching;
  const bool matched = try_maximum_bipartite_matching(edge_counts, matching);
  assert(matched && "Expected a perfect matching in conn cycle decomposition");
  return matching;
}

int directed_degree(const std::vector<std::vector<int>> &edge_counts, int node) {
  const int n = static_cast<int>(edge_counts.size());
  int degree = 0;
  for (int other = 0; other < n; ++other) {
    degree += edge_counts[node][other];
    degree += edge_counts[other][node];
  }
  return degree;
}

std::vector<std::vector<std::pair<int, int>>> decompose_into_perfect_matchings(
    const std::vector<std::vector<int>> &total_edge_cap,
    int expected_matchings) {
  std::vector<std::vector<int>> remaining = total_edge_cap;
  std::vector<std::vector<std::pair<int, int>>> matchings(expected_matchings);
  int num_matched = 0;

  while (directed_degree(remaining, 0) >= 1) {
    assert(num_matched < expected_matchings);
    const auto matching = maximum_bipartite_matching(remaining);
    for (const auto &edge : matching) {
      remaining[edge.first][edge.second]--;
    }
    matchings[num_matched] = matching;
    num_matched++;
  }

  assert(num_matched == expected_matchings);
  return matchings;
}

std::vector<std::vector<double>> sinkhorn_balance_demand_matrix(
    const std::vector<std::vector<double>> &demand) {
  const int n = static_cast<int>(demand.size());
  const double support_epsilon = 1e-9;
  const double tolerance = 1e-12;
  const int max_iterations = 10000;

  std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
  double total_demand = 0.0;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (row == col) {
        continue;
      }
      total_demand += demand[row][col];
    }
  }

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (row == col) {
        matrix[row][col] = 0.0;
        continue;
      }
      if (total_demand == 0.0) {
        matrix[row][col] = 1.0;
      } else {
        matrix[row][col] = std::max(demand[row][col], 0.0) + support_epsilon;
      }
    }
  }

  if (n <= 1) {
    return matrix;
  }

  for (int iter = 0; iter < max_iterations; ++iter) {
    for (int row = 0; row < n; ++row) {
      double row_sum = 0.0;
      for (int col = 0; col < n; ++col) {
        row_sum += matrix[row][col];
      }
      if (row_sum > 0.0) {
        for (int col = 0; col < n; ++col) {
          matrix[row][col] /= row_sum;
        }
      }
    }

    for (int col = 0; col < n; ++col) {
      double col_sum = 0.0;
      for (int row = 0; row < n; ++row) {
        col_sum += matrix[row][col];
      }
      if (col_sum > 0.0) {
        for (int row = 0; row < n; ++row) {
          matrix[row][col] /= col_sum;
        }
      }
    }

    double max_error = 0.0;
    for (int row = 0; row < n; ++row) {
      double row_sum = 0.0;
      for (int col = 0; col < n; ++col) {
        row_sum += matrix[row][col];
      }
      max_error = std::max(max_error, std::fabs(row_sum - 1.0));
    }
    for (int col = 0; col < n; ++col) {
      double col_sum = 0.0;
      for (int row = 0; row < n; ++row) {
        col_sum += matrix[row][col];
      }
      max_error = std::max(max_error, std::fabs(col_sum - 1.0));
    }

    if (max_error <= tolerance) {
      break;
    }
  }

  return matrix;
}

struct WeightedPermutation {
  double lambda = 0.0;
  std::vector<std::pair<int, int>> matching;
};

std::vector<WeightedPermutation> birkhoff_decompose_matrix(
    const std::vector<std::vector<double>> &doubly_stochastic_matrix) {
  const int n = static_cast<int>(doubly_stochastic_matrix.size());
  const double residual_tolerance = 1e-10;
  const int max_terms = std::max(1, n * n * 4);

  std::vector<std::vector<double>> remaining = doubly_stochastic_matrix;
  std::vector<WeightedPermutation> decomposition;

  for (int term_idx = 0; term_idx < max_terms; ++term_idx) {
    double max_residual = 0.0;
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < n; ++col) {
        if (remaining[row][col] >= 0.0 && remaining[row][col] < residual_tolerance) {
          remaining[row][col] = 0.0;
        }
        if (remaining[row][col] < 0.0 && std::fabs(remaining[row][col]) < residual_tolerance) {
          remaining[row][col] = 0.0;
        }
        assert(remaining[row][col] >= 0.0);
        max_residual = std::max(max_residual, remaining[row][col]);
      }
    }
    if (max_residual <= residual_tolerance) {
      break;
    }

    std::vector<std::vector<int>> support_graph(n, std::vector<int>(n, 0));
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < n; ++col) {
        if (row == col) {
          continue;
        }
        // Keep every positive residual edge in the support graph. Tiny edges are
        // still part of the valid Birkhoff support and can be necessary for the
        // next perfect matching after repeated subtraction.
        if (remaining[row][col] > 0.0) {
          support_graph[row][col] = 1;
        }
      }
    }

    std::vector<std::pair<int, int>> matching;
    const bool matched = try_maximum_bipartite_matching(support_graph, matching);
    assert(matched && "Expected a perfect matching in Birkhoff decomposition support graph");
    double lambda = std::numeric_limits<double>::max();
    for (const auto &edge : matching) {
      lambda = std::min(lambda, remaining[edge.first][edge.second]);
    }
    assert(lambda > 0.0);

    for (const auto &edge : matching) {
      remaining[edge.first][edge.second] -= lambda;
      if (remaining[edge.first][edge.second] < 0.0 &&
          std::fabs(remaining[edge.first][edge.second]) < residual_tolerance) {
        remaining[edge.first][edge.second] = 0.0;
      }
      if (remaining[edge.first][edge.second] < residual_tolerance) {
        remaining[edge.first][edge.second] = 0.0;
      }
    }

    WeightedPermutation term;
    term.lambda = lambda;
    term.matching = matching;
    decomposition.push_back(term);
  }

  double max_residual = 0.0;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      max_residual = std::max(max_residual, remaining[row][col]);
    }
  }
  assert(max_residual <= 1e-6);
  return decomposition;
}

simtime_picosec birkhoff_cycle_interval_from_demand(
    const std::vector<std::vector<double>> &demand,
    int alpha) {
  if (alpha <= 0) {
    return 0;
  }

  double max_row_sum_bytes = 0.0;
  double max_col_sum_bytes = 0.0;
  const int n = static_cast<int>(demand.size());
  for (int row = 0; row < n; ++row) {
    double row_sum_bytes = 0.0;
    double col_sum_bytes = 0.0;
    for (int col = 0; col < n; ++col) {
      row_sum_bytes += std::max(demand[row][col], 0.0);
      col_sum_bytes += std::max(demand[col][row], 0.0);
    }
    max_row_sum_bytes = std::max(max_row_sum_bytes, row_sum_bytes);
    max_col_sum_bytes = std::max(max_col_sum_bytes, col_sum_bytes);
  }

  const double bottleneck_bytes = std::max(max_row_sum_bytes, max_col_sum_bytes);
  if (bottleneck_bytes <= 0.0) {
    return 0;
  }

  const double aggregate_bps =
      static_cast<double>(alpha) * static_cast<double>(speedFromMbps((uint64_t)SPEED));
  if (aggregate_bps <= 0.0) {
    return 0;
  }

  const double cycle_interval_ps =
      bottleneck_bytes * 8.0 * 1.0e12 / aggregate_bps;
  simtime_picosec interval =
      static_cast<simtime_picosec>(std::llround(cycle_interval_ps));
  if (interval <= 0) {
    interval = 1;
  }
  return interval;
}

simtime_picosec vermilion_cycle_interval_from_demand(
    const std::vector<std::vector<double>> &demand,
    int alpha,
    int k) {
  if (alpha <= 0 || k <= 0) {
    return 0;
  }

  double max_row_sum_bytes = 0.0;
  double max_col_sum_bytes = 0.0;
  const int n = static_cast<int>(demand.size());
  for (int row = 0; row < n; ++row) {
    double row_sum_bytes = 0.0;
    double col_sum_bytes = 0.0;
    for (int col = 0; col < n; ++col) {
      row_sum_bytes += std::max(demand[row][col], 0.0);
      col_sum_bytes += std::max(demand[col][row], 0.0);
    }
    max_row_sum_bytes = std::max(max_row_sum_bytes, row_sum_bytes);
    max_col_sum_bytes = std::max(max_col_sum_bytes, col_sum_bytes);
  }

  const double bottleneck_bytes = std::max(max_row_sum_bytes, max_col_sum_bytes);
  if (bottleneck_bytes <= 0.0) {
    return 0;
  }

  const double aggregate_bps =
      static_cast<double>(alpha) * static_cast<double>(speedFromMbps((uint64_t)SPEED));
  if (aggregate_bps <= 0.0) {
    return 0;
  }

  const double scaled_bytes =
      bottleneck_bytes * (static_cast<double>(k + 1) / static_cast<double>(k));
  const double cycle_interval_ps =
      scaled_bytes * 8.0 * 1.0e12 / aggregate_bps;
  simtime_picosec interval =
      static_cast<simtime_picosec>(std::llround(cycle_interval_ps));
  if (interval <= 0) {
    interval = 1;
  }
  return interval;
}

std::vector<simtime_picosec> split_interval_evenly(
    simtime_picosec total_interval,
    size_t num_slots) {
  std::vector<simtime_picosec> intervals;
  if (total_interval <= 0 || num_slots == 0) {
    return intervals;
  }

  intervals.assign(num_slots, total_interval / static_cast<simtime_picosec>(num_slots));
  simtime_picosec remainder =
      total_interval % static_cast<simtime_picosec>(num_slots);
  for (size_t idx = 0; idx < num_slots; ++idx) {
    if (remainder <= 0) {
      break;
    }
    intervals[idx] += 1;
    remainder--;
  }
  for (size_t idx = 0; idx < num_slots; ++idx) {
    if (intervals[idx] <= 0) {
      intervals[idx] = 1;
    }
  }
  return intervals;
}

simtime_picosec interval_from_birkhoff_lambda(
    double lambda,
    simtime_picosec cycle_interval) {
  if (cycle_interval <= 0 || lambda <= 0.0) {
    return 0;
  }

  const double scaled_interval =
      lambda * static_cast<double>(cycle_interval);
  simtime_picosec interval =
      static_cast<simtime_picosec>(std::llround(scaled_interval));
  if (interval <= 0) {
    interval = 1;
  }
  return interval;
}

}  // namespace

// static bool has_active_alltoall_in_region(RegionalTopoManager* region_manager, int region_id);

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
    // tm.transform2symmetric(); // Keep the directional all-to-all matrix for Vermilion schedule generation.
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
      
      // removing the print for now, need to remove the call in the first place.
      // std::cout << "Traffic matrix for layer " << l << " region " << r << ":" << std::endl;
      // traffic_matrix[l][r]->print_mat();
    }
  }
}

RegionalTopoManager::RegionalTopoManager(Mixnet *topo, All2AllTrafficRecorder *demandrecorder, simtime_picosec refonc_delay, EventList &eventlist, int region_id, bool periodic_reconf, simtime_picosec periodic_interval, MixnetConnCycleAlgorithm conn_cycle_algorithm)
  : EventSource(eventlist, "MixnetRegionalTopoManager"), topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay), eventlist(eventlist), region_id(region_id), periodic_reconf(periodic_reconf), periodic_interval(periodic_interval), conn_cycle_algorithm(conn_cycle_algorithm), paused_flow_count(0), first_reconf_done(false)
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
  //std::cerr << "RegionalTopoManager::doNextEvent() status=" << static_cast<int>(status) << " time = " << eventlist.now() <<std::endl;
  if (status == TopoStatus::TOPO_LIVE)
  {
    

    if (periodic_reconf && !conn_cycle_ready)
    {
      // if(BVN==1)
      //   prepare_BVN();
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
      //std::cout<<"Stuck here for region "<< region_id << " non_empty_queues: " << non_empty_queues << std::endl;
      status = TopoStatus::TOPO_WAIT_DRAIN;
  }
  }
  else
  {
    finish_reconf();
    status = TopoStatus::TOPO_LIVE;
    reconfig_end_time = 0;
    //std::cout<< " Topology reconfiguration finished for region " << region_id << " at time " << eventlist.now() << std::endl;
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
    eventlist.sourceIsPendingRel(*this, timeFromNs(1)); // Deeksha: Check every 1 us if the queues are drained, to trigger the reconfiguration as soon as possible
    //std::cout<<"Stuck here for region "<< region_id << " non_empty_queues: " << non_empty_queues << std::endl;
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
      // std::cout << "reconfFired" << std::endl;
      // if (eq->queuesize() > 0)
      // {
      //   // eq->_state_send = LosslessQueue::PAUSE_RECEIVED;
      //   non_empty_queues += eq->_enqueued.size();
      //   // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      // }
      // else
      // {
      //   // eq->_state_send = LosslessQueue::PAUSED;
      // }
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
  //std::cout << "Region " << region_id <<" layer id " << current_layer_id << " reconfiguration will finish at " << reconfig_end_time << std::endl;
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
      // If a queue had a stale completion event from before the pause, drop it
      // before conditionally restarting service under the new topology.
      eventlist.cancelPendingSource(*eq);
      // std::cerr << "queue " << i << ", " << j << " br " << eq->_bitrate << " ps per byte " << eq->_ps_per_byte << " size " << eq->_enqueued.size() << std::endl;
      // vamsi: we can leave the queues non-empty, they just need to be paused, and resume when they have a connection
      // assert(eq->_enqueued.empty());      //Deeksha: Commented for testing 
      // if (eq->_bitrate > 0)
      {
        // Only resume the queues that have a connection available
        // eq->_state_send = LosslessQueue::READY;
        if (topo->conn[i][j] > 0)
        {
          eq->_state_send = LosslessQueue::READY;
          if (!eq->_enqueued.empty())
          {
            // eventlist.cancelPendingSource(*eq);
            eq->beginService();
            // std::cerr << "queue " << i << ", " << j << " br " << eq->_bitrate << " ps per byte " << eq->_ps_per_byte << " size " << eq->_enqueued.size() << std::endl;
            // std::cerr << "\t starting... " << std::endl;
          }
        }
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
  simtime_picosec next_periodic_interval = active_conn_cycle_interval;
  if (next_periodic_interval <= 0)
  {
    next_periodic_interval = periodic_interval;
  }
  if (periodic_reconf && next_periodic_interval > 0 &&
      (paused_flow_count > 0 || has_active_alltoall_in_region(this, region_id) || !first_reconf_done))
  {
    eventlist.sourceIsPending(*this, eventlist.now() + next_periodic_interval);
    //std::cout<< "periodic reconf scheduled at "<< region_id << current_layer_id << " at time " << eventlist.now() + next_periodic_interval << std::endl;
  }
  else
  {
    //std::cout<< "No periodic reconf scheduled at "<< region_id << current_layer_id << " at time " << eventlist.now() << " paused_flow_count: " << paused_flow_count << std::endl;
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

      // vamsi: We just need to pause the queue and allow pipes to drain (inflight bytes on the wire)

      eq->_state_send = LosslessQueue::PAUSED;
      // Queue backlog should stay queued across reconfiguration. Cancel any
      // already-scheduled dequeue so only packets already inside the pipe drain.
      eventlist.cancelPendingSource(*eq);

      // if(eq->queuesize() > 0)
      // {
      //   std::cout << "Queue" << eq->_enqueued.size()
      //             << " state=" << eq->_state_send << std::endl;
      //   // Allow existing backlog to drain; flows are paused so no new packets enter
      //   // Treat both PAUSED and PAUSE_RECEIVED as needing an explicit resume. // vamsi
      //   if (eq->_state_send == LosslessQueue::PAUSED ||
      //       eq->_state_send == LosslessQueue::PAUSE_RECEIVED) {
      //     eq->_state_send = LosslessQueue::READY;
      //     if (!eq->_enqueued.empty()) {
      //       eq->beginService();
      //     }
      //   }

      //   non_empty_queues += eq->_enqueued.size();
      //   // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      // }
      
      Pipe *p = topo->pipes[i][j];
      if (!p->_inflight.empty()){
        // while (!p->_inflight.empty())
        // {
        //   p->_inflight.pop_front();
        // }

        //Do nothing, just counting the inflight packets
        //std::cerr << "Draining pipe " << i << ", " << j << " size " << p->_inflight.size() <<std::endl;
        //std::cout<< "*Check*" << std::endl;

        // vamsi: only account inflight packets; don't change queue state here
        non_empty_queues += p->_inflight.size();
        //std::cout<<"Inflight packet types in pipe p->type "<< p->type << std::endl;
       // std::cout<<"Pipe "<<i<<","<<j<<" size "<<p->_inflight.size()<< std::endl;
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
      
      
      //std::cerr << "Pausing flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " at index " << conn_cycle_idx << std::endl;
      if(tcpsrc->is_all2all && tcpsrc->is_elec)
      {
        //std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is electric." << std::endl;
      }
      else if(tcpsrc->is_all2all && !tcpsrc->is_elec)
      {
        //std::cout << "Flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " is optical." << std::endl;
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
  //std::cout << "For region " << region_id << " Total flows paused: " << paused_flow_count << std::endl;
}


bool RegionalTopoManager::use_rotor_paths() const
{
  return conn_cycle_algorithm == CONN_CYCLE_ROTOR;
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
      assert(tcpsrc->is_all2all);

      if (use_rotor_paths())
      {
        // Rotor two-hop uses the complete-graph two-hop route set carried by
        // packets. It does not need per-slot route rewrites here.
        i++;
        continue;
      }

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

      // vamsi: We don't need to update routes with periodic reconfiguration: Assumption is that a path is available over time. We just need to fix direct paths, and use when available
      // Route *routeout, *routein;
      // int choice;
      // vector<const Route *> *srcpaths = topo->get_paths(tcpsrc->_flow_src, tcpsrc->_flow_dst);
      // // choice = rand()%srcpaths->size(); // comment this out if we want to use the first path
      // choice = 0; // Deeksha: Choosing the first path
      // routeout = new Route(*(srcpaths->at(choice)));
      // routeout->push_back(tcpsrc->_sink);
      // vector<const Route *> *dstpaths = topo->get_paths(tcpsrc->_flow_dst, tcpsrc->_flow_src);
      // // choice = rand()%dstpaths->size(); // comment this out if we want to use the first path
      // choice = 0; // Deeksha: Choosing the first path
      // routein = new Route(*(dstpaths->at(choice)));
      // routein->push_back(tcpsrc);
      // tcpsrc->update_route(routeout, routein);

      // delete srcpaths;
      // delete dstpaths;
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
      bool can_resume = false;
      if (use_rotor_paths())
      {
        can_resume = has_mixnet_two_hop_path(topo, tcpsrc->_flow_src, tcpsrc->_flow_dst);
      }
      else
      {
        // Only resume flows that have an active OCS connection in the current conn matrix.
        can_resume = topo->conn[src_node][dst_node] > 0;
      }
      if (can_resume)
      {
        // vamsi: clear regional pause flag and resume the flow
        tcpsrc->resume_flow_in_region();
        tcpsrc->resume_flow();
        //std::cerr << "Resuming flow from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << " at index " << conn_cycle_idx << std::endl;
        //paused_flow_count--;
        resumed_flows++;
      }
      else
      {
        //std::cout << "Keeping flow paused (no OCS conn) from " << tcpsrc->_flow_src << " to " << tcpsrc->_flow_dst << std::endl;
      }
    }
    i++;
  }
  //std::cout << "For region " << region_id << " Total flows resumed: " << resumed_flows << " out of " << paused_flow_count <<std::endl;
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
        topo->queues[i+start_node][j+start_node]->_bitrate = (reconfig_conn_matrix[i][j]>1? reconfig_conn_matrix[i][j] : 1) * speedFromMbps((uint64_t)SPEED);
        topo->queues[i+start_node][j+start_node]->_ps_per_byte = (simtime_picosec)((pow(10.0, 12.0) * 8) / topo->queues[i+start_node][j+start_node]->_bitrate);
      }
    }
  }
 //Printing the updated conn matrix for debugging
  // std::cout << "Updated conn matrix for region " << region_id << " at layer " << current_layer_id << ":" << std::endl;
  // for (int i = 0; i < region_size; i++) {
  //   for (int j = 0; j < region_size; j++) {
  //     std::cout << reconfig_conn_matrix[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
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


void RegionalTopoManager::compute_schedule_vermilion()
{
  conn_cycle.clear();
  conn_cycle_intervals.clear();
  // Vermilion
  const int N = region_size;
  const int dE = alpha;
  const int k = 4;
  const int period = (k * N) / dE;
  const int scaled_degree = (k - 1) * N;
  const int expected_matchings = k * N;

  assert(N % 2 == 0);
  assert(dE > 0);
  assert((k * N) % dE == 0);

  const Matrix2D<double> &stored_tm =
      *demandrecorder->traffic_matrix[current_layer_id][region_id];
  std::vector<std::vector<double>> saturated_demand =
      load_demand_matrix_for_conn_cycle(stored_tm, N, reverse != 0);
  simtime_picosec vermilion_cycle_interval =
      vermilion_cycle_interval_from_demand(saturated_demand, dE, k);
  if (vermilion_cycle_interval <= 0)
  {
    vermilion_cycle_interval =
        periodic_interval * static_cast<simtime_picosec>(period);
  }
  std::vector<std::vector<double>> normalized_demand =
      normalize_demand_matrix(saturated_demand);
  std::vector<std::vector<int>> rounded_matrix =
      floor_scaled_demand_matrix(normalized_demand, scaled_degree);

  std::vector<int> out_left(N, 0);
  std::vector<int> in_left(N, 0);
  for (int row = 0; row < N; ++row) {
    int row_sum = 0;
    int col_sum = 0;
    for (int col = 0; col < N; ++col) {
      row_sum += rounded_matrix[row][col];
      col_sum += rounded_matrix[col][row];
    }
    out_left[row] = scaled_degree - row_sum;
    in_left[row] = scaled_degree - col_sum;
    assert(out_left[row] >= 0);
    assert(in_left[row] >= 0);
  }

  std::vector<std::vector<int>> config_counts =
      directed_configuration_model_counts(in_left, out_left, rounded_matrix);
  std::vector<std::vector<int>> total_edge_cap(
      N, std::vector<int>(N, 0));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        total_edge_cap[i][j] =
            rounded_matrix[i][j] + config_counts[i][j] + 1;
      }
    }
  }

  // The +1 off-diagonal base gives each row/column only (N - 1) extra units.
  // Add one more fixed derangement matching so total_edge_cap becomes exactly
  // kN-regular and the packed conn cycle does not leave one matching slot empty.
  for (int i = 0; i < N; ++i) {
    const int j = (i + 1) % N;
    total_edge_cap[i][j] += 1;
  }

  for (int row = 0; row < N; ++row) {
    int row_sum = 0;
    int col_sum = 0;
    for (int col = 0; col < N; ++col) {
      row_sum += total_edge_cap[row][col];
      col_sum += total_edge_cap[col][row];
    }
    assert(row_sum == expected_matchings);
    assert(col_sum == expected_matchings);
  }

  std::vector<std::vector<std::pair<int, int>>> matchings =
      decompose_into_perfect_matchings(total_edge_cap, expected_matchings);
  std::vector<std::vector<std::vector<int>>> adjacency(
      period, std::vector<std::vector<int>>(N, std::vector<int>(N, 0)));

  for (int timeslot = 0; timeslot < period; ++timeslot) {
    for (int index = 0; index < dE; ++index) {
      const int matching_idx = timeslot * dE + index;
      if (matching_idx >= static_cast<int>(matchings.size())) {
        continue;
      }
      for (const auto &edge : matchings[matching_idx]) {
        adjacency[matching_idx % period][edge.first][edge.second] += 1;
      }
    }
  }

  std::vector<simtime_picosec> slot_intervals =
      split_interval_evenly(vermilion_cycle_interval, adjacency.size());
  if (slot_intervals.empty())
  {
    slot_intervals.assign(adjacency.size(), periodic_interval);
  }

  for (int slot = 0; slot < period; ++slot) {
    for (int row = 0; row < N; ++row) {
      int row_sum = 0;
      int col_sum = 0;
      for (int col = 0; col < N; ++col) {
        row_sum += adjacency[slot][row][col];
        col_sum += adjacency[slot][col][row];
      }
      assert(row_sum == alpha);
      assert(col_sum == alpha);
    }
    conn_cycle.push_back(adjacency[slot]);
    conn_cycle_intervals.push_back(slot_intervals[slot]);
  }

  conn_cycle_ready = !conn_cycle.empty();
  conn_cycle_idx = 0;
  active_conn_cycle_interval = conn_cycle_intervals.empty() ? periodic_interval
                                                            : conn_cycle_intervals[0];
  // std::cout << "Conn cycle initialized from demand matrix with "
  //           << conn_cycle.size()
  //           << " connection matrices and "
  //           << conn_cycle_intervals.size()
  //           << " interval entries."
  //           << " cycle_interval_ps " << vermilion_cycle_interval
  //           << " algorithm " << conn_cycle_algorithm_name(conn_cycle_algorithm)
  //           << " reverse: " << reverse
  //           << " layer " << current_layer_id
  //           << " region " << region_id
  //           << " time " << eventlist.now() << std::endl;
}

void RegionalTopoManager::compute_schedule_birkhoff()
{
  conn_cycle.clear();
  conn_cycle_intervals.clear();

  const int N = region_size;
  const Matrix2D<double> &stored_tm =
      *demandrecorder->traffic_matrix[current_layer_id][region_id];
  std::vector<std::vector<double>> demand =
      load_demand_matrix_for_conn_cycle(stored_tm, N, reverse != 0);
  simtime_picosec birkhoff_cycle_interval =
      birkhoff_cycle_interval_from_demand(demand, alpha);
  if (birkhoff_cycle_interval <= 0)
  {
    birkhoff_cycle_interval = periodic_interval;
  }
  std::vector<std::vector<double>> doubly_stochastic_matrix =
      sinkhorn_balance_demand_matrix(demand);
  std::vector<WeightedPermutation> decomposition =
      birkhoff_decompose_matrix(doubly_stochastic_matrix);

  for (const auto &term : decomposition) {
    std::vector<std::vector<int>> conn(
        N, std::vector<int>(N, 0));
    for (const auto &edge : term.matching) {
      conn[edge.first][edge.second] = alpha;
    }
    conn_cycle.push_back(conn);
  }

  for (const auto &term : decomposition) {
    conn_cycle_intervals.push_back(
        interval_from_birkhoff_lambda(term.lambda, birkhoff_cycle_interval));
  }

  conn_cycle_ready = !conn_cycle.empty();
  conn_cycle_idx = 0;
  active_conn_cycle_interval = conn_cycle_intervals.empty() ? periodic_interval
                                                            : conn_cycle_intervals[0];
  // std::cout << "Conn cycle initialized from demand matrix with "
  //           << conn_cycle.size()
  //           << " connection matrices and "
  //           << conn_cycle_intervals.size()
  //           << " interval entries."
  //           << " cycle_interval_ps " << birkhoff_cycle_interval
  //           << " algorithm " << conn_cycle_algorithm_name(conn_cycle_algorithm)
  //           << " reverse: " << reverse
  //           << " layer " << current_layer_id
  //           << " region " << region_id
  //           << " time " << eventlist.now() << std::endl;
}

// Deeksha: Update conn cycle as per the selected adjacency matrix generation algorithm
void RegionalTopoManager::prepare_conn_cycle()
{ 
  if (!demandrecorder->check_traffic_matrix(current_layer_id, region_id))
  {
    return;
  }

  switch (conn_cycle_algorithm)
  {
  case CONN_CYCLE_VERMILION:
    compute_schedule_vermilion();
    return;
  case CONN_CYCLE_ROTOR:
    rotor_conn_cycle();
    return;
  case CONN_CYCLE_BIRKHOFF:
    compute_schedule_birkhoff();
    return;
  default:
    compute_schedule_vermilion();
    return;
  }

//   // vamsi: period is k * n/u
//   int k = 5;
//   int connsize = k * region_size/alpha;
//   std::vector<std::vector<std::vector<int>>> conn(connsize,
//     std::vector<std::vector<int>>(region_size,
//         std::vector<int>(region_size, 0)));
// if(reverse==0){
// conn={
// {
// {0,1,1,0,1,0,1,0},
// {1,0,0,1,0,1,1,0},
// {0,0,0,1,0,1,0,2},
// {1,0,1,0,0,1,1,0},
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,2,0,0,0},
// {1,1,0,1,0,0,0,1},
// {0,1,1,0,1,0,1,0}
// },
// {
// {0,1,1,0,1,0,1,0},
// {1,0,0,1,0,1,1,0},
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,0,1,1,0},
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,1,0,0,1},
// {0,1,0,1,1,0,0,1},
// {1,0,1,0,1,0,1,0}
// },
// {
// {0,1,1,0,1,0,1,0},
// {1,0,0,1,0,1,1,0},
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,0,1,1,0},
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,1,0,0,1},
// {0,1,0,1,1,0,0,1},
// {1,0,1,0,1,0,1,0}
// },
// {
// {0,1,0,0,1,0,1,1},
// {2,0,0,1,0,1,0,0},
// {0,0,0,1,1,1,0,1},
// {1,0,1,0,0,0,1,1},
// {0,1,0,0,0,2,1,0},
// {1,0,1,1,1,0,0,0},
// {0,1,1,1,0,0,0,1},
// {0,1,1,0,1,0,1,0}
// },
// {
// {0,1,0,0,1,0,1,1},
// {2,0,0,1,1,0,0,0},
// {0,0,0,1,1,2,0,0},
// {0,0,1,0,0,0,1,2},
// {1,0,0,0,0,2,1,0},
// {1,1,0,1,1,0,0,0},
// {0,1,2,0,0,0,0,1},
// {0,1,1,1,0,0,1,0}
// },
// {
// {0,1,1,0,1,1,0,0},
// {1,0,0,1,1,0,0,1},
// {0,0,0,1,1,1,0,1},
// {1,0,0,0,1,0,0,2},
// {0,1,1,0,0,0,2,0},
// {1,0,1,1,0,0,1,0},
// {0,1,1,1,0,1,0,0},
// {1,1,0,0,0,1,0,0}
// },
// {
// {0,1,0,1,1,0,1,0},
// {0,0,1,1,1,0,0,1},
// {1,0,0,1,0,1,1,0},
// {1,1,0,0,1,1,0,0},
// {0,0,1,0,0,0,1,1},
// {0,2,0,0,0,0,1,1},
// {1,0,1,0,1,1,0,0},
// {1,0,1,1,0,1,0,0}
// },
// {
// {0,1,0,1,1,0,1,0},
// {0,0,1,1,0,1,0,1},
// {1,0,0,1,1,0,1,0},
// {1,1,0,0,1,1,0,0},
// {0,0,1,0,0,0,1,1},
// {0,2,0,0,0,0,1,1},
// {1,0,1,0,1,1,0,0},
// {1,0,1,1,0,1,0,0}
// },
// {
// {0,1,0,1,0,1,1,0},
// {0,0,1,1,0,1,1,0},
// {1,0,0,0,2,0,0,1},
// {0,2,0,0,0,1,1,0},
// {1,0,2,0,0,0,1,0},
// {0,1,1,1,0,0,0,1},
// {0,0,0,0,2,0,0,1},
// {2,0,0,1,0,1,0,0}
// },
// {
// {0,1,0,1,0,2,0,0},
// {0,0,1,0,2,0,1,0},
// {1,0,0,0,1,0,1,0},
// {1,1,0,0,0,1,0,1},
// {0,0,2,0,0,0,1,0},
// {0,1,0,2,0,0,0,1},
// {1,0,0,0,1,0,0,1},
// {1,1,0,1,0,1,0,0}
// },
// {
// {0,0,1,1,0,1,0,1},
// {0,0,1,0,2,0,0,1},
// {2,1,0,0,0,0,0,0},
// {0,2,0,0,1,0,0,1},
// {1,0,0,1,0,0,2,0},
// {0,0,0,1,0,0,1,1},
// {0,0,1,0,1,1,0,0},
// {1,1,0,1,0,1,0,0}
// },
// {
// {0,0,1,1,0,1,0,1},
// {0,0,1,0,1,0,0,1},
// {2,1,0,0,0,0,0,0},
// {0,1,0,0,1,0,1,0},
// {1,0,0,1,0,0,1,1},
// {0,0,0,1,0,0,1,1},
// {0,0,1,0,1,1,0,0},
// {1,1,0,0,0,1,0,0}
// },
// {
// {0,0,1,0,1,1,0,1},
// {0,0,0,1,1,0,0,1},
// {0,1,0,0,0,0,2,0},
// {0,1,0,0,1,0,0,1},
// {2,0,1,1,0,0,0,0},
// {0,0,0,1,0,0,1,1},
// {1,0,1,0,0,1,0,0},
// {1,1,0,0,0,1,0,0}
// },
// {
// {0,0,1,0,1,1,0,1},
// {0,0,0,1,1,0,0,1},
// {0,1,0,0,0,0,2,0},
// {0,1,0,0,1,0,0,1},
// {2,0,1,1,0,0,0,0},
// {0,0,0,1,0,0,1,1},
// {2,0,1,0,0,0,0,0},
// {0,1,0,0,0,2,0,0}
// },
// {
// {0,0,1,0,1,0,1,0},
// {0,0,0,1,0,1,0,1},
// {1,1,0,0,0,1,0,0},
// {0,0,0,0,1,0,1,1},
// {0,0,1,1,0,0,0,1},
// {1,1,0,0,0,0,1,0},
// {1,1,0,1,0,0,0,0},
// {0,0,1,0,1,1,0,0}
// },
// {
// {0,0,1,0,1,0,0,1},
// {0,0,0,1,0,1,1,0},
// {0,0,0,0,0,1,1,1},
// {1,0,0,0,0,1,1,0},
// {0,1,0,1,0,0,0,1},
// {1,0,1,0,1,0,0,0},
// {1,1,0,1,0,0,0,0},
// {0,1,1,0,1,0,0,0}
// }
// };

// }
// else{
//   conn={
// {
// {0,1,1,0,0,0,1,1},
// {1,0,0,0,1,1,0,1},
// {0,1,0,1,0,1,0,1},
// {0,1,1,0,1,0,1,0},
// {1,0,1,1,0,1,0,0},
// {0,1,0,1,1,0,1,0},
// {1,0,1,0,0,1,0,1},
// {1,0,0,1,1,0,1,0}
// },
// {
// {0,1,1,0,0,0,1,1},
// {1,0,0,0,1,1,0,1},
// {0,0,0,2,0,1,0,1},
// {0,1,1,0,1,0,1,0},
// {0,1,0,2,0,1,0,0},
// {1,0,1,0,1,0,1,0},
// {1,1,0,0,0,1,0,1},
// {1,0,1,0,1,0,1,0}
// },
// {
// {0,1,1,0,0,1,0,1},
// {1,0,0,0,1,1,0,1},
// {0,0,0,2,1,0,0,1},
// {1,1,1,0,0,0,1,0},
// {0,1,0,2,0,1,0,0},
// {0,1,1,0,1,0,1,0},
// {1,0,0,0,0,1,0,1},
// {1,0,1,0,1,0,1,0}
// },
// {
// {0,1,0,1,1,1,0,0},
// {1,0,1,0,0,1,0,1},
// {0,0,0,1,1,0,1,1},
// {1,0,1,0,1,0,0,1},
// {0,2,0,1,0,1,0,0},
// {0,0,1,0,1,0,1,0},
// {0,1,0,1,0,1,0,1},
// {2,0,1,0,0,0,1,0}
// },
// {
// {0,1,0,1,1,1,0,0},
// {1,0,1,0,0,1,0,1},
// {1,0,0,1,1,0,1,0},
// {1,0,1,0,1,0,0,1},
// {0,1,0,0,0,1,0,1},
// {0,0,1,1,1,0,1,0},
// {0,1,0,1,0,1,0,1},
// {1,1,1,0,0,0,1,0}
// },
// {
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,0,1,0,1},
// {1,0,0,1,1,0,1,0},
// {2,0,1,0,1,0,0,0},
// {0,1,0,0,0,1,0,1},
// {0,0,1,1,1,0,1,0},
// {0,1,0,1,0,1,0,1},
// {0,1,1,0,1,0,1,0}
// },
// {
// {0,1,0,1,0,1,0,1},
// {0,0,1,0,0,0,1,2},
// {0,1,0,1,1,1,0,0},
// {2,0,1,0,0,1,0,0},
// {0,1,0,0,0,0,1,1},
// {1,0,1,1,1,0,0,0},
// {0,1,0,1,1,1,0,0},
// {1,0,1,0,1,0,1,0}
// },
// {
// {0,1,0,1,0,1,0,1},
// {1,0,1,0,1,0,1,0},
// {0,1,0,1,0,1,0,0},
// {1,0,1,0,0,1,0,1},
// {1,1,0,0,0,0,1,1},
// {1,0,1,0,0,0,0,1},
// {0,1,0,1,2,0,0,0},
// {0,0,1,0,1,1,1,0}
// },
// {
// {0,1,0,1,0,0,1,1},
// {0,0,2,0,0,0,1,0},
// {0,0,0,1,1,0,1,1},
// {0,0,0,0,1,2,0,1},
// {1,1,0,1,0,0,0,1},
// {2,0,0,0,0,0,1,0},
// {0,0,1,1,1,1,0,0},
// {1,1,0,0,1,0,0,0}
// },
// {
// {0,0,1,1,0,0,1,1},
// {0,0,0,1,1,0,1,0},
// {1,0,0,0,0,1,1,0},
// {1,1,0,0,0,1,0,1},
// {1,0,2,0,0,0,1,0},
// {0,1,0,0,0,0,0,2},
// {1,0,0,1,1,0,0,0},
// {0,1,0,1,1,1,0,0}
// },
// {
// {0,0,1,1,1,0,1,0},
// {0,0,0,1,1,0,1,0},
// {1,0,0,0,0,1,1,0},
// {1,1,0,0,0,1,0,1},
// {0,0,2,0,0,0,1,0},
// {1,0,0,0,0,0,0,2},
// {1,0,0,1,1,0,0,0},
// {0,2,0,1,0,1,0,0}
// },
// {
// {0,0,1,1,1,0,1,0},
// {0,0,0,1,1,0,1,0},
// {1,0,0,0,0,1,1,0},
// {0,1,0,0,0,1,0,1},
// {1,0,1,1,0,0,1,0},
// {1,0,0,0,0,0,0,2},
// {1,0,1,0,1,0,0,0},
// {0,2,0,1,0,1,0,0}
// },
// {
// {0,0,1,0,2,0,1,0},
// {0,0,0,1,0,1,1,0},
// {1,1,0,0,0,1,0,0},
// {0,1,0,0,0,0,1,1},
// {2,0,1,0,0,0,1,0},
// {0,1,0,1,0,0,0,1},
// {0,0,1,0,1,0,0,1},
// {1,0,0,1,0,1,0,0}
// },
// {
// {0,0,1,0,1,1,1,0},
// {0,0,0,1,1,1,0,0},
// {1,1,0,0,0,0,1,0},
// {0,1,0,0,0,0,1,1},
// {1,0,0,0,0,0,1,1},
// {1,1,0,1,0,0,0,1},
// {1,0,1,0,1,0,0,0},
// {0,0,1,1,0,1,0,0}
// },
// {
// {0,0,1,0,1,1,0,1},
// {0,0,0,1,1,1,0,0},
// {1,1,0,0,0,0,1,0},
// {0,1,0,0,0,0,1,1},
// {1,0,0,0,0,0,1,1},
// {1,1,0,1,0,0,0,1},
// {1,0,1,0,1,0,0,0},
// {0,0,1,1,0,1,0,0}
// },
// {
// {0,0,1,0,1,0,1,0},
// {0,0,0,1,0,1,0,1},
// {1,1,0,0,0,1,0,0},
// {0,1,0,0,1,0,1,0},
// {1,0,0,0,0,0,0,2},
// {0,1,0,1,0,0,1,0},
// {1,0,1,0,1,0,0,0},
// {0,0,1,1,0,1,0,0}
// }
// };
// }


//   // conn = {
//   //   {// conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    },
//   //   {  // conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    },
//   //    {// conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    },
//   //   {  // conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    },
     
//   //   { // conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    },
//   //   { // conn[5]
//   //    {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1},
//   //      {1,1,1,1,1,1,1,1},
//   //     {1,1,1,1,1,1,1,1}
//   //    } 
//   //   };

//   // conn = {
//   //   { // conn[0]
//   //     {1,1,2,1,1,0,1,0},
//   //     {1,1,0,2,0,1,2,0},
//   //     {2,0,1,1,0,1,0,1},
//   //     {1,2,1,1,0,1,0,0},
//   //     {1,0,0,0,2,0,1,1},
//   //     {0,1,1,1,0,1,1,2},
//   //     {1,2,0,0,1,1,1,1},
//   //     {0,0,1,0,1,2,1,1}
//   //   },
//   //   { // conn[1]
//   //     {1,0,1,1,0,1,1,1},
//   //     {0,2,0,1,0,1,1,1},
//   //     {1,0,1,1,1,1,1,1},
//   //     {1,1,1,1,1,2,0,0},
//   //     {0,0,1,1,1,1,1,1},
//   //     {1,1,1,2,1,1,0,0},
//   //     {1,1,1,0,1,0,0,2},
//   //     {1,1,1,0,1,0,2,0}
//   //   },
//   //   { // conn[2]
//   //     {0,1,1,1,1,0,1,1},
//   //     {1,0,0,1,0,1,1,1},
//   //     {1,0,1,1,1,2,0,1},
//   //     {1,1,1,0,1,1,1,0},
//   //     {1,0,1,1,0,1,1,1},
//   //     {0,1,2,1,1,0,1,0},
//   //     {1,1,0,1,1,1,0,1},
//   //     {1,1,1,0,1,0,1,1}
//   //   },
//   //   { // conn[3]
//   //     {0,1,0,1,1,1,1,1},
//   //     {1,0,1,1,1,1,1,0},
//   //     {0,1,0,1,2,0,1,0},
//   //     {1,1,1,0,0,0,1,1},
//   //     {1,1,2,0,1,0,0,1},
//   //     {1,1,0,0,0,0,1,1},
//   //     {1,1,1,1,0,1,0,0},
//   //     {1,0,0,1,1,1,0,1}
//   //   },
//   //   { // conn[4]
//   //     {0,1,0,1,1,1,0,2},
//   //     {1,0,1,0,1,1,1,1},
//   //     {0,1,0,0,2,0,1,1},
//   //     {1,0,0,0,1,1,2,1},
//   //     {1,1,2,1,0,0,0,1},
//   //     {1,1,0,1,0,0,1,1},
//   //     {0,1,1,2,0,1,0,0},
//   //     {2,1,1,1,1,1,0,0}
//   //   },
//   //   { // conn[5]
//   //     {0,1,1,1,0,1,0,1},
//   //     {1,0,1,0,1,1,2,1},
//   //     {1,1,0,1,0,1,1,1},
//   //     {1,0,1,0,2,1,0,1},
//   //     {0,1,0,2,0,0,2,1},
//   //     {1,1,1,1,0,0,1,1},
//   //     {0,2,1,0,2,1,0,0},
//   //     {1,1,1,1,1,1,0,0}
//   //   }
//   // };

//   // conn = {
//   //   { // conn[0]
//   //     {1,0,0,1,1,0,1,0},
//   //     {0,1,1,0,0,1,0,1},
//   //     {1,1,1,0,0,1,0,0},
//   //     {0,1,1,1,0,0,0,1},
//   //     {1,0,0,0,2,0,1,0},
//   //     {0,1,1,0,0,1,1,0},
//   //     {0,0,0,2,0,1,1,0},
//   //     {1,0,0,0,1,0,0,2}
//   //   },
 
//   //   { // conn[2]
//   //     {1,0,0,1,1,0,1,0},
//   //     {0,1,1,0,1,1,0,0},
//   //     {0,0,1,1,0,0,0,2},
//   //     {1,0,0,0,1,1,1,0},
//   //     {0,1,0,1,0,1,1,0},
//   //     {1,0,0,0,1,1,0,1},
//   //     {1,1,1,0,0,0,0,1},
//   //     {0,1,1,1,0,0,1,0}
//   //   },
    
//   //   { // conn[4]
//   //     {0,1,0,1,1,0,1,0},
//   //     {1,0,1,0,1,0,1,0},
//   //     {0,0,1,1,0,0,0,2},
//   //     {1,0,0,0,1,1,1,0},
//   //     {0,1,0,1,0,1,0,1},
//   //     {1,0,1,0,1,1,0,0},
//   //     {1,1,1,0,0,0,0,1},
//   //     {0,1,0,1,0,1,1,0}
//   //   },
    
//   //   { // conn[6]
//   //     {0,1,0,0,1,1,0,1},
//   //     {1,0,0,1,1,0,1,0},
//   //     {0,1,1,1,0,1,0,0},
//   //     {1,0,1,0,1,0,0,1},
//   //     {0,1,1,1,0,0,1,0},
//   //     {1,0,0,0,0,0,2,1},
//   //     {0,0,1,0,1,2,0,0},
//   //     {1,1,0,1,0,0,0,1}
//   //   },
    
//   //   { // conn[8]
//   //     {0,1,0,0,1,1,1,0},
//   //     {1,0,0,1,1,0,0,1},
//   //     {0,1,1,0,0,1,1,0},
//   //     {0,1,1,0,0,1,1,0},
//   //     {1,0,1,0,1,0,0,1},
//   //     {0,0,1,2,0,0,0,1},
//   //     {2,0,0,1,0,0,0,1},
//   //     {0,1,0,0,1,1,1,0}
//   //   },
    
//   //   { // conn[10]
//   //     {0,0,1,0,1,1,0,1},
//   //     {0,1,0,1,0,0,1,1},
//   //     {1,0,0,0,1,0,2,0},
//   //     {1,0,1,0,0,2,0,0},
//   //     {1,1,1,0,1,0,0,0},
//   //     {0,1,0,2,0,0,0,1},
//   //     {1,1,0,1,0,0,0,1},
//   //     {0,0,1,0,1,1,1,0}
//   //   },
    
//   // };
// //Deeksha: This is for reverse
//   // conn = {
//   //   { // conn[0]
//   //     {1,1,0,1,0,1,0,0},
//   //     {0,1,0,0,1,0,1,1},
//   //     {1,0,1,0,2,0,0,0},
//   //     {1,0,1,1,0,1,0,0},
//   //     {0,0,1,1,1,0,0,1},
//   //     {0,0,1,0,0,1,1,1},
//   //     {1,1,0,1,0,0,1,0},
//   //     {0,1,0,0,0,1,1,1}
//   //   },
//   //   { // conn[1]
//   //     {1,0,1,0,0,1,1,0},
//   //     {0,1,0,0,1,1,0,1},
//   //     {0,1,1,0,0,0,0,2},
//   //     {1,1,0,1,1,0,0,0},
//   //     {0,1,0,1,1,0,1,0},
//   //     {0,0,2,0,0,1,1,0},
//   //     {1,0,0,1,1,0,1,0},
//   //     {1,0,0,1,0,1,0,1}
//   //   },
//   //   { // conn[2]
//   //     {1,0,1,1,0,0,1,0},
//   //     {0,1,0,1,0,1,0,1},
//   //     {0,0,1,0,1,1,1,0},
//   //     {0,1,1,0,1,0,0,1},
//   //     {1,0,0,1,0,1,0,1},
//   //     {1,1,0,1,0,0,1,0},
//   //     {1,0,0,0,2,1,0,0},
//   //     {0,1,1,0,0,0,1,1}
//   //   },
//   //   { // conn[3]
//   //     {0,0,0,1,1,0,1,1},
//   //     {1,1,0,1,0,1,0,0},
//   //     {0,0,0,1,1,1,0,1},
//   //     {1,0,1,0,0,0,2,0},
//   //     {0,1,1,0,0,1,0,1},
//   //     {1,0,1,0,1,0,0,1},
//   //     {0,1,1,1,0,0,1,0},
//   //     {1,1,0,0,1,1,0,0}
//   //   },
//   //   { // conn[4]
//   //     {1,0,0,1,1,0,0,1},
//   //     {0,0,2,1,0,1,0,0},
//   //     {1,1,0,0,0,1,1,0},
//   //     {1,0,0,0,1,0,0,2},
//   //     {0,1,0,1,0,0,2,0},
//   //     {0,1,1,0,1,0,1,0},
//   //     {0,1,1,1,0,0,0,1},
//   //     {1,0,0,0,1,2,0,0}
//   //   },
//   //   { // conn[5]
//   //     {0,1,0,1,1,0,0,1},
//   //     {1,0,1,0,0,1,1,0},
//   //     {2,0,0,1,0,1,0,0},
//   //     {0,0,2,0,0,0,1,1},
//   //     {1,0,0,0,0,1,1,1},
//   //     {0,2,0,0,2,0,0,0},
//   //     {0,1,1,1,0,0,0,1},
//   //     {0,0,0,1,1,1,1,0}
//   //   }
//   // };
//   // conn = {
//   //   { // conn[0]
//   //     {1,0,1,0,1,0,1,0},
//   //     {0,1,1,0,1,0,1,0},
//   //     {1,1,1,0,0,1,0,0},
//   //     {0,1,0,1,1,0,0,1},
//   //     {0,1,0,0,1,1,1,0},
//   //     {1,0,0,1,0,1,0,1},
//   //     {0,0,0,2,0,0,1,1},
//   //     {1,0,1,0,0,1,0,1}
//   //   },
//   //   { // conn[1]
//   //     {1,0,1,0,0,1,1,0},
//   //     {0,1,0,1,0,1,0,1},
//   //     {1,0,1,1,0,0,0,1},
//   //     {1,1,0,1,0,0,1,0},
//   //     {0,1,0,0,1,1,1,0},
//   //     {0,0,1,0,1,1,0,1},
//   //     {1,0,0,1,1,0,1,0},
//   //     {0,1,1,0,1,0,0,1}
//   //   },
//   //   { // conn[2]
//   //     {1,0,0,1,0,1,1,0},
//   //     {0,0,1,1,1,0,0,1},
//   //     {1,1,1,0,0,0,0,1},
//   //     {0,0,0,1,0,1,1,1},
//   //     {1,1,0,0,1,1,0,0},
//   //     {0,1,0,0,1,1,1,0},
//   //     {0,1,1,0,1,0,0,1},
//   //     {1,0,1,1,0,0,1,0}
//   //   },
//   //   { // conn[3]
//   //     {0,1,0,1,0,1,0,1},
//   //     {2,0,0,1,1,0,0,0},
//   //     {1,0,1,0,1,0,1,0},
//   //     {0,0,1,0,1,1,0,1},
//   //     {1,0,1,1,0,0,0,1},
//   //     {0,1,0,0,0,1,2,0},
//   //     {0,1,0,1,0,1,0,1},
//   //     {0,1,1,0,1,0,1,0}
//   //   },
//   //   { // conn[4]
//   //     {0,1,0,1,0,0,1,1},
//   //     {2,0,0,1,0,1,0,0},
//   //     {0,0,1,1,2,0,0,0},
//   //     {1,0,0,0,1,0,1,1},
//   //     {1,1,1,0,0,1,0,0},
//   //     {0,1,0,1,0,0,1,1},
//   //     {0,0,1,0,0,2,0,1},
//   //     {0,1,1,0,1,0,1,0}
//   //   },
//   //   { // conn[5]
//   //     {0,0,1,0,1,0,1,1},
//   //     {0,0,1,0,1,1,1,0},
//   //     {1,1,0,1,0,1,0,0},
//   //     {1,1,1,0,1,0,0,0},
//   //     {0,0,0,0,0,1,2,1},
//   //     {1,0,1,1,0,0,0,1},
//   //     {0,2,0,1,0,0,0,1},
//   //     {1,0,0,1,1,1,0,0}
//   //   }
//   // };
// // conn = {
// //     { // conn[0]
// //       {1,1,1,0,0,1,0,0},
// //       {0,1,1,1,0,0,1,0},
// //       {1,0,1,0,1,0,1,0},
// //       {1,0,0,1,1,1,0,0},
// //       {0,0,0,1,1,0,1,1},
// //       {0,1,1,0,0,1,0,1},
// //       {1,1,0,0,0,0,1,1},
// //       {0,0,0,1,1,1,0,1}
// //     },
// //     { // conn[1]
// //       {0,2,0,1,0,1,0,0},
// //       {0,0,1,0,1,0,1,1},
// //       {1,0,1,0,1,0,0,1},
// //       {0,0,0,1,0,1,1,1},
// //       {0,1,1,1,1,0,0,0},
// //       {0,1,0,1,0,1,0,1},
// //       {1,0,1,0,1,0,1,0},
// //       {2,0,0,0,0,1,1,0}
// //     },
// //     { // conn[2]
// //       {0,2,0,0,1,0,0,1},
// //       {0,0,0,1,0,1,1,1},
// //       {1,0,1,0,1,0,0,1},
// //       {2,0,0,1,0,1,0,0},
// //       {0,1,1,1,1,0,0,0},
// //       {0,0,1,1,0,0,1,1},
// //       {0,1,1,0,1,1,0,0},
// //       {1,0,0,0,0,1,2,0}
// //     },
// //     { // conn[3]
// //       {0,0,1,0,1,1,0,1},
// //       {1,0,0,1,0,1,1,0},
// //       {0,1,1,0,1,0,1,0},
// //       {1,1,0,1,0,0,0,1},
// //       {0,0,0,0,0,2,1,1},
// //       {0,1,1,0,1,0,0,1},
// //       {1,0,1,1,0,0,1,0},
// //       {1,1,0,1,1,0,0,0}
// //     },
// //     { // conn[4]
// //       {0,0,1,0,1,0,1,1},
// //       {1,0,0,1,1,1,0,0},
// //       {0,1,0,1,0,1,1,0},
// //       {1,1,1,0,0,0,0,1},
// //       {1,0,0,0,0,2,0,1},
// //       {1,0,1,1,0,0,1,0},
// //       {0,1,0,1,1,0,0,1},
// //       {0,1,1,0,1,0,1,0}
// //     },
// //     { // conn[5]
// //       {0,0,1,0,0,1,1,1},
// //       {1,0,0,1,2,0,0,0},
// //       {0,0,0,1,0,1,1,1},
// //       {0,0,0,0,1,1,1,1},
// //       {2,1,1,0,0,0,0,0},
// //       {1,0,1,1,0,0,1,0},
// //       {0,2,0,0,0,1,0,1},
// //       {0,1,1,1,1,0,0,0}
// //     }
// //   };
//   //std::vector<int> avail_conn(region_size, alpha);
//   for (int k = 0; k < conn.size(); k++)  //Deeksha :CHeck this later
//   {
//     // for(int i = 0; i < region_size; i++) {
//     //   for(int j = 0; j < region_size; j++) {
//     //     // Initialize connection matrix
//     //       conn[k][j][i] = conn[k][i][j];       
//     //   }
//     // }
        
//     conn_cycle.push_back(conn[k]);
//     conn_cycle_intervals.push_back(periodic_interval);
//   }
  
//   conn_cycle_ready = !conn_cycle.empty();
//   conn_cycle_idx = 0;
//   active_conn_cycle_interval = periodic_interval;
//   std::cout << "Conn cycle initialized with " << conn_cycle.size() << " connection matrices." << " reverse: " << reverse << " layer "<< current_layer_id << "region " << region_id << " time " << eventlist.now() << std::endl;
}

//Deeksha: Adding the function here to find if there is any active all-to-all flow in the region, to decide whether we can skip the reconfiguration or not
bool RegionalTopoManager::has_active_alltoall_in_region(RegionalTopoManager* region_manager, int region_id) {  
    // list<TcpSrc *>::iterator i = region_manager->demandrecorder->rtx_scanner->_tcps.begin();  
    // while (i != region_manager->demandrecorder->rtx_scanner->_tcps.end()) {  
    //     TcpSrc* tcpsrc = *i;  
          
    //     // Check if it's an all-to-all flow and in the specified region  
    //     if (tcpsrc->is_all2all && region_manager->is_in_region(tcpsrc)) {  
    //         return true; // Found active all-to-all in this region  
    //     }  
    //     i++;  
    // }  
    // return false; // No active all-to-all found in this region  
  return activeAlltoAll;
}



void RegionalTopoManager::rotor_conn_cycle()
{ 
  if (!demandrecorder->check_traffic_matrix(current_layer_id, region_id))
  {
    return;
  }

  conn_cycle.clear();
  conn_cycle_intervals.clear();
  const int N = region_size;
  assert(alpha > 0);
  assert(N > 0);
  assert(N % alpha == 0);

  const int cycle_size = N / alpha;
  std::vector<std::vector<int>> coverage(
      N, std::vector<int>(N, 0));

  // std::vector<int> avail_conn(region_size, alpha);
  for (int slot = 0; slot < cycle_size; slot++)
  {
    std::vector<std::vector<int>> conn(
        N, std::vector<int>(N, 0));

    // Build one alpha-regular bipartite matching union using alpha cyclic shifts.
    // Across all slots, the shifts cover 0..N-1 exactly once, so the sum of the
    // conn matrices is the all-ones matrix.
    for (int offset_in_slot = 0; offset_in_slot < alpha; offset_in_slot++)
    {
      const int shift = slot * alpha + offset_in_slot;
      for (int i = 0; i < N; i++)
      {
        const int j = (i + shift) % N;
        conn[i][j] = 1;
        coverage[i][j]++;
      }
    }

    // for (int i = 0; i < region_size; i++) {
    //   for (int j = 0; j < region_size; j++) {
    //     if (i == j) {
    //       continue; // No self-connections
    //     }
    //     else if(j == (i+k)%region_size){
    //       conn[i][j] = 1; // Initialize connection matrix
    //       conn[j][i] = 1;
    //       break;
    //     }
    //   }
    // }
    conn_cycle.push_back(conn);
    conn_cycle_intervals.push_back(periodic_interval);
  }

  for (int slot = 0; slot < cycle_size; slot++)
  {
    for (int i = 0; i < N; i++)
    {
      int row_sum = 0;
      int col_sum = 0;
      for (int j = 0; j < N; j++)
      {
        row_sum += conn_cycle[slot][i][j];
        col_sum += conn_cycle[slot][j][i];
      }
      assert(row_sum == alpha);
      assert(col_sum == alpha);
    }
  }

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      assert(coverage[i][j] == 1);
    }
  }
  
  conn_cycle_ready = !conn_cycle.empty();
  conn_cycle_idx = 0;
  active_conn_cycle_interval = periodic_interval;
  // std::cout << "Rotor conn cycle initialized with "
  //           << conn_cycle.size()
  //           << " connection matrices."
  //           << " cycle_size " << cycle_size
  //           << " alpha " << alpha
  //           << " reverse: " << reverse
  //           << " layer " << current_layer_id
  //           << " region " << region_id
  //           << " time " << eventlist.now() << std::endl;
}


std::vector<std::vector<int>> RegionalTopoManager::regional_topo_reconfig() {

  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  if (conn_cycle_idx < conn_cycle_intervals.size())
  {
    active_conn_cycle_interval = conn_cycle_intervals[conn_cycle_idx];
  }
  else
  {
    active_conn_cycle_interval = periodic_interval;
  }
  conn = conn_cycle[conn_cycle_idx];         // Updating the regional conn matrix from the conn cycle
  conn_cycle_idx++;
  conn_cycle_idx = (conn_cycle_idx) % conn_cycle.size(); // Deeksha: Rotate to next conn matrix
  
  // update the conn of the topo
  // Debug print for conn matrix
  // std::cerr << "Debug - Conn matrix for region " << region_id << " layer " << current_layer_id << ":" << "conn index: " << conn_cycle_idx << "conn matrix size: " << conn_cycle.size() << " region_size: " << region_size << std::endl;
  // for (int i = 0; i < region_size; i++) {
  //   for (int j = 0; j < region_size; j++) {
  //     std::cerr << conn[i][j] << " ";
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << "Available connections: ";
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      // if (!reverse){
        if(conn[i][j] > 0) {
          topo->conn[i+start_node][j+start_node] = conn[i][j];
          // vamsi: Oh no! We should set the reverse connection, that can override what we need for i-j, also this is done in mixnet because they assume birectional links and construct graph based on this assumption.
          // topo->conn[j+start_node][i+start_node] = conn[i][j];
        }
        else {
          topo->conn[i+start_node][j+start_node] = 0;
          // topo->conn[j+start_node][i+start_node] = 0;
        }
      // }
      // else{
      //   if(conn[i][j] > 0) {
      //     topo->conn[j+start_node][i+start_node] = conn[i][j];
      //   }
      //   else {
      //     topo->conn[j+start_node][i+start_node] = 0;
      //   }

      // }
    }
  }
  
  return conn;
}

std::vector<std::vector<int>> RegionalTopoManager::rotor_reconfig() {

  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  if (conn_cycle_idx < conn_cycle_intervals.size())
  {
    active_conn_cycle_interval = conn_cycle_intervals[conn_cycle_idx];
  }
  else
  {
    active_conn_cycle_interval = periodic_interval;
  }
  conn = conn_cycle[conn_cycle_idx];         // Updating the regional conn matrix from the conn cycle
  conn_cycle_idx = (conn_cycle_idx + 1) % conn_cycle.size(); // Deeksha: Rotate to next conn matrix
  
  // update the conn of the topo
  // Debug print for conn matrix
  // std::cerr << "Debug - Conn matrix for region " << region_id << " layer " << current_layer_id << ":" << std::endl;
  // for (int i = 0; i < region_size; i++) {
  //   for (int j = 0; j < region_size; j++) {
  //     std::cerr << conn[i][j] << " ";
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << "Available connections: ";
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

MixnetTopoManager::MixnetTopoManager(Mixnet *topo, All2AllTrafficRecorder *demandrecorder, simtime_picosec refonc_delay, EventList &eventlist, bool periodic_reconf, simtime_picosec periodic_interval, MixnetConnCycleAlgorithm conn_cycle_algorithm)
    : topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay), periodic_reconf(periodic_reconf), periodic_interval(periodic_interval), conn_cycle_algorithm(conn_cycle_algorithm)
{

  region_num = topo->region_num;
  region_size = topo->region_size;
  nnodes = topo->no_of_nodes();
  alpha = topo->alpha;

  // init the regional topo managers
  for (int i = 0; i < region_num; i++)
  {
    regional_topo_managers.push_back(new RegionalTopoManager(topo, demandrecorder, reconf_delay, eventlist, i, periodic_reconf, periodic_interval, conn_cycle_algorithm));
  }
}
