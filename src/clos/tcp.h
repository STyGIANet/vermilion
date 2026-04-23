// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-        
#ifndef TCP_H
#define TCP_H

/*
 * A TCP source and sink
 */

#include <list>
#include "config.h"
#include "network.h"
#include "tcppacket.h"
#include "eventlist.h"
#include "sent_packets.h"
#include <utility>
//#include "dyn_net_sch.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <stack>
#include <algorithm>
#include <vector>
#include <cmath>
// #define MODEL_RECEIVE_WINDOW 1

#define timeInf 0

// #define PACKET_SCATTER 1
// #define RANDOM_PATH 1

//#define MAX_SENT 10000
template< class T >
class Matrix2D;
struct DemandRecorder;

class TcpSink;
class MultipathTcpSrc;
class MultipathTcpSink;


template <typename dtype>
struct Element {
    size_t row;
    size_t col;
    dtype value;

    Element(size_t r, size_t c, dtype v) : row(r), col(c), value(v) {}
};

template< class T >
std::ostream &operator<<( std::ostream &, const Matrix2D< T > & );

template< class dtype >
class Matrix2D {
 public:
  Matrix2D() : n_cols(0), n_rows(0), mat(nullptr), base(0) {}  // 添加默认构造函数
  Matrix2D( size_t n_cols, size_t n_rows ) : n_cols( n_cols ), n_rows( n_rows ), mat( nullptr ) {
    mat = new dtype *[n_rows];
    for ( size_t row = 0; row < n_rows; row ++ )
      mat[ row ] = new dtype[n_cols];
    fill_zeros( );
    base=0;// zero -> init
  }

  virtual ~Matrix2D( ) {
    for ( size_t row = 0; row < n_rows; row ++ )
      delete[] mat[ row ];
    delete[] mat;
  }

//(syj): comment this if there are some mistake
//   Matrix2D( const Matrix2D & ) = delete;
//   Matrix2D &operator=( const Matrix2D & ) = delete;

 private:
  size_t n_cols;
  size_t n_rows;
  dtype **mat;
  size_t base;
 public:
  void fill_zeros( );
  void update_base(size_t new_base);
  dtype get_elem( size_t row, size_t col ) const;
  size_t get_base() const { return base; }

  void set_elem( size_t row, size_t col, const dtype value );

  void add_by( const Matrix2D< dtype > &matrix_2_d );

  void sub_by( const Matrix2D< dtype > &matrix_2_d );

  void mul_by( const dtype &value );

  void add_elem_by( size_t row, size_t col, const dtype value );

  void sub_elem_by( size_t row, size_t col, const dtype value );

  void copy_from( const Matrix2D< dtype > &matrix_2_d );

  void normalize_by_max( );

  void normalize( ); //Deeksha: Added normalize function

  void floor_mat( ); //Deeksha: Added floor function

  void print_mat( ); //Deeksha: Added print function

  std::vector<std::pair<size_t, size_t>> get_sorted_indices();// return (row,col) pairs based on the value

  void downsamplefrom(const Matrix2D<dtype>& traffic_matrix, int region);

  void set_row(size_t row_idx, size_t value);

  void set_col(size_t col_idx, size_t value);

  bool check_eq(const Matrix2D<dtype>& other) const {
    if (n_rows != other.n_rows || n_cols != other.n_cols) {
      return false;
    }
    for (size_t row = 0; row < n_rows; row++) {
      for (size_t col = 0; col < n_cols; col++) {
        if (mat[row][col] != other.get_elem(row, col)) {
          return false;
        }
      }
    }
    return true;
  }

  bool is_symmetric() const {
    if (n_rows != n_cols) {
      return false;  // 非方阵不可能是对称矩阵
    }
    for (size_t i = 0; i < n_rows; i++) {
      for (size_t j = i + 1; j < n_cols; j++) {  // 只需要检查上三角部分
        if (mat[i][j] != mat[j][i]) {
          return false;
        }
      }
    }
    return true;
  }

  void transform2symmetric() {
    if (n_rows != n_cols) {
      throw std::runtime_error("Cannot transform non-square matrix to symmetric");
    }
    // 遍历上三角部分
    for (size_t i = 0; i < n_rows; i++) {
      for (size_t j = i + 1; j < n_cols; j++) {
        dtype sum_ = mat[i][j] + mat[j][i];
        mat[i][j] = sum_;
        mat[j][i] = sum_;
      }
    }
  }

//     // 可移动构造函数
//   Matrix2D(Matrix2D&& other) noexcept
//       : n_cols(other.n_cols), n_rows(other.n_rows), mat(other.mat), base(other.base) {
//     other.mat = nullptr;
//     other.n_cols = 0;
//     other.n_rows = 0;
//   }

//   // 可移动赋值运算符
//   Matrix2D& operator=(Matrix2D&& other) noexcept {
//     if (this != &other) {
//       for (size_t row = 0; row < n_rows; row++)
//         delete[] mat[row];
//       delete[] mat;

//       n_cols = other.n_cols;
//       n_rows = other.n_rows;
//       mat = other.mat;
//       base = other.base;

//       other.mat = nullptr;
//       other.n_cols = 0;
//       other.n_rows = 0;
//     }
//     return *this;
//   }

  friend std::ostream &operator
  <<< dtype >(
  std::ostream &os,
  const Matrix2D< dtype > &d
  );
};

template<class dtype>
void Matrix2D< dtype >::set_row(size_t row_idx, size_t value) {
  for(size_t col = 0; col < n_cols; col++) {
    mat[row_idx][col] = value;
  }
}

template<class dtype>
void Matrix2D< dtype >::set_col(size_t col_idx, size_t value) {
  for(size_t row = 0; row < n_rows; row++) {
    mat[row][col_idx] = value;
  }
}


template<class dtype>
std::vector<std::pair<size_t, size_t>> Matrix2D< dtype >::get_sorted_indices() {
    std::vector<Element<dtype>> elements;
    elements.reserve(n_rows * n_cols);  // 预先分配空间，避免动态扩容
    // 遍历整个矩阵并将每个元素(row, col, value)保存到elements中
    for (size_t row = 0; row < n_rows; row++) {
        for (size_t col = 0; col < n_cols; col++) {
            elements.emplace_back(row, col, mat[row][col]);
        }
    }

    // 按照value值从大到小对elements排序
    std::sort(elements.begin(), elements.end(), [](Element<dtype>& a, Element<dtype>& b) {
        return a.value > b.value;
    });

    // 构建并返回排序后的(row, col) pair的向量
    std::vector<std::pair<size_t, size_t>> sorted_indices;
    sorted_indices.reserve(elements.size());  // 预先分配空间
    for (const auto& elem : elements) {
        sorted_indices.emplace_back(elem.row, elem.col);
    }

    return sorted_indices;
}

template<class dtype>
void Matrix2D< dtype >::downsamplefrom(const Matrix2D<dtype>& traffic_matrix, int region) {
    // 遍历新的矩阵
    for (size_t new_row = 0; new_row < n_rows; ++new_row) {
        for (size_t new_col = 0; new_col < n_cols; ++new_col) {
            dtype sum = 0;
            // 计算原矩阵中对应的8x8区域的和
            for (size_t i = 0; i < region; ++i) {
                for (size_t j = 0; j < region; ++j) {
                    sum += traffic_matrix.get_elem(new_row * region + i, new_col * region + j);
                }
            }
            // 设置新矩阵中的元素
            mat[new_row][new_col] = sum;
        }
    }
}

template< class dtype >
void Matrix2D< dtype >::fill_zeros( ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] = 0;
  }
}

template< class dtype >
void Matrix2D< dtype >::add_by( const Matrix2D< dtype > &matrix_2_d ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ ) {
      mat[ row ][ col ] += matrix_2_d.get_elem( row, col );
    }
  }
}

template< class dtype >
void Matrix2D< dtype >::sub_by( const Matrix2D< dtype > &matrix_2_d ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] -= matrix_2_d.get_elem( row, col );
  }
}

template< class dtype >
void Matrix2D< dtype >::mul_by( const dtype &value ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] *= value;
  }
}

template< class dtype >
dtype Matrix2D< dtype >::get_elem( size_t row, size_t col ) const {
  return mat[ row ][ col ];
}

template< class dtype >
void Matrix2D< dtype >::update_base( size_t new_base ){
  // if( base > new_base){
  //   base = new_base;
  // }
  base = new_base;
}

template< class dtype >
void Matrix2D< dtype >::copy_from( const Matrix2D< dtype > &matrix_2_d ) {
//  if ( matrix_2_d.n_cols > n_cols || matrix_2_d.n_rows > n_rows ) {
//    throw std::runtime_error( "matrix dimensions do not match for copying." );
//  }
  size_t min_n_rows = std::min( matrix_2_d.n_rows, n_rows );
  size_t min_n_cols = std::min( matrix_2_d.n_cols, n_cols );
  for ( size_t row = 0; row < min_n_rows; row ++ ) {
    for ( size_t col = 0; col < min_n_cols; col ++ )
      mat[ row ][ col ] = matrix_2_d.get_elem( row, col ); //todo:use memcpy
  }
}

template< class dtype >
void Matrix2D< dtype >::normalize_by_max( ) {
  dtype max_data = std::numeric_limits< dtype >::min( );
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
        if ( row != col )
            max_data = ( max_data < mat[ row ][ col ] ? mat[ row ][ col ] : max_data );
  }
  if ( max_data != 0 ) {
    for ( size_t row = 0; row < n_rows; row ++ ) {
      for ( size_t col = 0; col < n_cols; col ++ ) {
        mat[ row ][ col ] /= max_data;
        //mat[ row ][ col ] = double( int( 100. * mat[ row ][ col ] ) ) / 100.;
        mat[ row ][ col ] = double( mat[ row ][ col ] );
      }
    }
  }
}

//Deeksha: Added normalize function
template< class dtype >
void Matrix2D< dtype >::normalize( ) {
    dtype max_row_sum = std::numeric_limits< dtype >::min( );;
    dtype max_col_sum = std::numeric_limits< dtype >::min( );;

    // Compute max row sum
    for (size_t row = 0; row < n_rows; row++) {
        dtype row_sum = 0;
        for (size_t col = 0; col < n_cols; col++) {
            row_sum += mat[row][col];
        }
        if (row_sum > max_row_sum)
            max_row_sum = row_sum;
    }

    // Compute max column sum
    for (size_t col = 0; col < n_cols; col++) {
        dtype col_sum = 0;
        for (size_t row = 0; row < n_rows; row++) {
            col_sum += mat[row][col];
        }
        if (col_sum > max_col_sum)
            max_col_sum = col_sum;
    }

    dtype max_sum = std::max(max_row_sum, max_col_sum);

    // Normalize matrix
    if (max_sum != dtype(0)) {
        for (size_t row = 0; row < n_rows; row++) {
            for (size_t col = 0; col < n_cols; col++) {
                mat[row][col] /= max_sum;
            }
        }
    }
}

//Deeksha: Creating a floor function
template< class dtype >
void Matrix2D< dtype >::floor_mat() {
    for (size_t row = 0; row < n_rows; row++) {
        for (size_t col = 0; col < n_cols; col++) {
            mat[row][col] = std::floor(mat[row][col]);
        }
    }
    std::cout << "After flooring the matrix: " << std::endl;
}


template< class dtype >
void Matrix2D< dtype >::print_mat() {
    for (size_t row = 0; row < n_rows; row++) {
        for (size_t col = 0; col < n_cols; col++) {
            std::cout << "mat[" << row << "][" << col << "] = " << mat[row][col] << " ";
        }
        std::cout << std::endl;
    }
}
template< class dtype >
void Matrix2D< dtype >::set_elem( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] = value;
}

template< class dtype >
void Matrix2D< dtype >::add_elem_by( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] += value;
}

template< class dtype >
void Matrix2D< dtype >::sub_elem_by( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] -= value;
}

template< class dtype >
std::ostream &operator<<( std::ostream &os, const Matrix2D< dtype > &d ) {
  for ( size_t row = 0; row < d.n_rows; row ++ ) {
    for ( size_t col = 0; col < d.n_cols; col ++ )
      os << d.mat[ row ][ col ] << " ";
    os << std::endl;
  }
  return os;
}

struct tcp_info {
    /*
    we can get flow size, src, dst from flowsrc
    */
    enum Type {
        all2all_groupby_forward,
        all2all_groupby_backward,
        all2all_aggregate_forward,
        all2all_aggregate_backward,
        allreduce,
        p2p
    };
    Type tcp_type;
    int micro_batch_id;
    int layer_id;
    int target_micro_batch_id;//only used in p2p task
    int target_layer_id;//only used in p2p task
    int allreduce_idx;//for ringallreduce task
    int operatorsize;

    tcp_info(const std::string& info, int mb_id, int l_id, int t_mb_id = 0, int t_l_id = 0, int ar_idx = 0, int op_size = 0)
        : micro_batch_id(mb_id), 
          layer_id(l_id), 
          target_micro_batch_id(t_mb_id), 
          target_layer_id(t_l_id), 
          allreduce_idx(ar_idx), 
          operatorsize(op_size) 
    {
        if (info == "GROUP_BY forward all2all") {
            tcp_type = all2all_groupby_forward;
        } else if (info == "GROUP_BY backward all2all") {
            tcp_type = all2all_groupby_backward;
        } else if (info == "AGGREGATE forward all2all") {
            tcp_type = all2all_aggregate_forward;
        } else if (info == "AGGREGATE backward all2all") {
            tcp_type = all2all_aggregate_backward;
        } else {
            assert(false);
        }
    }
    tcp_info(Type t, int mb_id, int l_id, int t_mb_id = 0, int t_l_id = 0, int ar_idx = 0, int op_size = 0)
        : micro_batch_id(mb_id), 
          layer_id(l_id), 
          target_micro_batch_id(t_mb_id), 
          target_layer_id(t_l_id), 
          allreduce_idx(ar_idx), 
          operatorsize(op_size) 
    {
        tcp_type = t;
    }

};
class TcpSrc : public PacketSink, public EventSource {
    friend class TcpSink;
 public:
   
    TcpSrc(TcpLogger* logger, TrafficLogger* pktlogger, ofstream * _fstream_out, 
            EventList &eventlist, int flow_src, int flow_dst, 
            void (*acf)(void*) = nullptr, void* acd = nullptr);
    ~TcpSrc();
    uint32_t get_id(){ return id;}
    virtual void connect(const Route& routeout, const Route& routeback, 
			 TcpSink& sink, simtime_picosec startTime);
    void startflow();
    inline void joinMultipathConnection(MultipathTcpSrc* multipathSrc) {
	_mSrc = multipathSrc;
    };


    void (*application_callback)(void*);
    void * application_callback_data;

    static DemandRecorder * demand_recorder;

    void doNextEvent();
    virtual void receivePacket(Packet& pkt);

    void replace_route(const Route* newroute);

    void set_flowsize(uint64_t flow_size_in_bytes);

    void set_ssthresh(uint64_t s){_ssthresh = s;}

    uint32_t effective_window();
    virtual void rtx_timer_hook(simtime_picosec now,simtime_picosec period);
    virtual const string& nodename() { return _nodename; }

    inline uint64_t get_flowsize() {return _flow_size;} // bytes
    inline int get_flow_src() {return _flow_src;}
    inline int get_flow_dst() {return _flow_dst;}
    inline void set_start_time(simtime_picosec startTime) {_start_time = startTime;}
    inline simtime_picosec get_start_time() {return _start_time;};

    static void pause_flow();
    void pause_flow_in_region();
    static void resume_all_flow();
    void resume_flow_in_region();
    
    void update_route(Route* routeout, Route *routein);
    void resume_flow();

    // should really be private, but loggers want to see:
    uint64_t _highest_sent;  //seqno is in bytes
    uint64_t _packets_sent;
    uint64_t _flow_size;
    uint32_t _cwnd;
    uint32_t _maxcwnd;
    uint64_t _last_acked;
    uint32_t _ssthresh;
    uint32_t _dupacks;
    bool _finished;
    uint32_t DUPACK_TH;
    uint16_t _crt_path;
    ofstream * fstream_out;

    int32_t _app_limited;
    
    static bool tcp_flow_paused;
    bool tcp_flow_paused_in_region = false;
    bool tcp_has_pending_send = false;
    bool tcp_has_pending_retrans = false;


    //round trip time estimate, needed for coupled congestion control
    simtime_picosec _rtt, _rto, _mdev,_base_rtt;
    int _cap;
    simtime_picosec _rtt_avg, _rtt_cum;
    //simtime_picosec when[MAX_SENT];
    int _sawtooth;

    uint16_t _mss;
    uint32_t _unacked; // an estimate of the amount of unacked data WE WANT TO HAVE in the network
    uint32_t _effcwnd; // an estimate of our current transmission rate, expressed as a cwnd
    uint64_t _recoverq;
    bool _in_fast_recovery;

    bool _established;

    uint32_t _drops;

    TcpSink* _sink;
    MultipathTcpSrc* _mSrc;
    simtime_picosec _RFC2988_RTO_timeout;
    bool _rtx_timeout_pending;

    void set_app_limit(int pktps);

    const Route* _route;
    simtime_picosec _last_ping;
    vector<const Route*>* _paths;
    bool _scatter_random_paths = false;

    void set_paths(vector<const Route*>* rt, bool randomize = false);
    void clear_paths();
    void send_packets();

	
#ifdef MODEL_RECEIVE_WINDOW
    SentPackets _sent_packets;
    uint64_t _highest_data_seq;
#endif
    int _subflow_id;

    virtual void inflate_window();
    virtual void deflate_window();

    simtime_picosec _start_time;
    int _flow_src; // the sender (source) for this flow
    int _flow_dst; // the receiver (sink) for this flow

    bool is_elec = false; // we use it to identify the flow in elec fabric
    bool is_all2all = false; // we use it to identify the flow from all2all task
 private:
    const Route* _old_route;
    uint64_t _last_packet_with_old_route;

    // Housekeeping
    TcpLogger* _logger;
    //TrafficLogger* _pktlogger;

    // Connectivity
    PacketFlow _flow;

    // Mechanism
    void clear_timer(uint64_t start,uint64_t end);

    void retransmit_packet();
    //simtime_picosec _last_sent_time;

    //void clearWhen(TcpAck::seq_t from, TcpAck::seq_t to);
    //void showWhen (int from, int to);
    string _nodename;
};

class TcpSink : public PacketSink, public DataReceiver, public Logged {
    friend class TcpSrc;
 public:
    TcpSink();
    ~TcpSink();

    inline void joinMultipathConnection(MultipathTcpSink* multipathSink){
	_mSink = multipathSink;
    };

    void receivePacket(Packet& pkt);
    TcpAck::seq_t _cumulative_ack; // the packet we have cumulatively acked
    uint64_t _packets;
    uint32_t _drops;
    uint64_t cumulative_ack(){ return _cumulative_ack + _received.size()*1000;}
    uint32_t drops(){ return _src->_drops;}
    uint32_t get_id(){ return id;}
    virtual const string& nodename() { return _nodename; }

    MultipathTcpSink* _mSink;
    list<TcpAck::seq_t> _received; /* list of packets above a hole, that 
				      we've received */

    vector<const Route*>* _paths;
    bool _scatter_random_paths = false;

    void set_paths(vector<const Route*>* rt, bool randomize = false);
    void clear_paths();

    TcpSrc* _src;
 private:
    // Connectivity
    uint16_t _crt_path;

    void connect(TcpSrc& src, const Route& route);
    const Route* _route;

    // Mechanism
    void send_ack(simtime_picosec ts,bool marked);

    string _nodename;
};

// struct MicrobatchLayerKey {
//     int microbatch;
//     int layer;

//     bool operator==(const MicrobatchLayerKey &other) const {
//         return microbatch == other.microbatch && layer == other.layer;
//     }
// };
// struct MicrobatchLayerKeyHash {
//     std::size_t operator()(const MicrobatchLayerKey &key) const {
//         return std::hash<int>()(key.microbatch) ^ (std::hash<int>()(key.layer) << 1);
//     }
// };


class TcpRtxTimerScanner : public EventSource {
 public:
    TcpRtxTimerScanner(simtime_picosec scanPeriod, EventList& eventlist);
    TcpRtxTimerScanner(simtime_picosec scanPeriod, EventList& eventlist,int enable_tcppairs,int micro_batch_num, int layer_num, int n_cols,int n_rows);
    TcpRtxTimerScanner(simtime_picosec scanPeriod, EventList &eventlist, int enable_tcppairs);
    void doNextEvent();
    void registerTcp(TcpSrc &tcpsrc);
    void registerTcp(TcpSrc &tcpsrc,tcp_info &tcpinfo);
//  private:
    simtime_picosec _scanPeriod;
    typedef list<TcpSrc*> tcps_t;
    tcps_t _tcps;

    //(syj) add task level info for tcp task
    typedef pair<TcpSrc*, tcp_info*> TcpPair;
    typedef list<TcpPair> TcpPairList;
    TcpPairList _tcppairs;
    std::vector<std::vector<Matrix2D<double>>> traffic_matrix; // [microbatch, layer, Matrix2D]
    
    //std::vector<int> microbatch;
    int _enable_tcppairs;
};

#endif
