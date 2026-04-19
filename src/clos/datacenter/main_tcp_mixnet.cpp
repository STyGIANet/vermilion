// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "config.h"
#include <sstream>
#include <strstream>
#include <fstream> // need to read flows
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <unistd.h>
#include "network.h"
#include "randomqueue.h"
//#include "shortflows.h"
#include "pipe.h"
#include "eventlist.h"
#include "logfile.h"
#include "loggers.h"
#include "clock.h"
#include "tcp.h"
#include "mtcp.h"
#include "compositequeue.h"
#include "firstfit.h"
#include "topology.h"
#include "dyn_net_sch.h"
//#include "connection_matrix.h"

// Choose the topology here:
// #include "test_topology.h"
#include "flat_topology.h"
#include "fat_tree_topology.h"
#include "ffapp.h"
#include "mixnet.h"
#include "mixnet_topomanager.h"
#include <list>

// Simulation params

#define PRINT_PATHS 0

#define PERIODIC 0
#include "main.h"

uint32_t RTT = 1000; // ns
int DEFAULT_NODES = 16;

uint32_t SPEED;
uint32_t RTT_rack = 500; // ns
uint32_t RTT_net = 500;  // ns
std::ofstream fct_util_out_ecs;
std::ofstream fct_util_out_ocs;
//std::ofstream fct_util_out_ocs;
FirstFit *ff = NULL;
//unsigned int subflow_count = 8; // probably not necessary ???

#define DEFAULT_PACKET_SIZE 9000 // full packet (including header), Bytes
#define DEFAULT_HEADER_SIZE 64   // header size, Bytes
#define DEFAULT_QUEUE_SIZE 100
#define DEFAULT_DEGREE 4
#define DEFAULT_RECONF_DELAY 100

string ntoa(double n);
string itoa(uint64_t n);

EventList eventlist;
// Logfile* lg;

void exit_error(char *progr, char *param)
{
  cerr << "Bad parameter: " << param << endl;
  cerr << "Usage " << progr << " [UNCOUPLED(DEFAULT)|COUPLED_INC|FULLY_COUPLED|COUPLED_EPSILON] [epsilon][COUPLED_SCALABLE_TCP]" << endl;
  exit(1);
}

void print_path(std::ofstream &paths, const Route *rt)
{
  for (unsigned int i = 1; i < rt->size() - 1; i += 2)
  {
    RandomQueue *q = (RandomQueue *)rt->at(i);
    if (q != NULL)
      paths << q->str() << " ";
    else
      paths << "NULL ";
  }

  paths << endl;
}

std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(now_tm, "%m-%d-%H-%M-%S");
    return oss.str();
}

std::string getExecutableName() {
    char buffer[1024];
    ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer));
    if (count != -1) {
        std::filesystem::path execPath(std::string(buffer, count));
        return execPath.filename().string();
    } else {
        // 处理读取错误的情况
        std::cerr << "Error reading the executable path." << std::endl;
        return "";
    }
}
int find_nonum(int _no_nodes){
  int k=0;
  while( k*k*k/4 < _no_nodes){
    k+=2;
  }
  return k*k*k/4;
}
int main(int argc, char **argv)
{

  TcpPacket::set_packet_size(DEFAULT_PACKET_SIZE - DEFAULT_HEADER_SIZE); // MTU
  mem_b queuesize = DEFAULT_QUEUE_SIZE * DEFAULT_PACKET_SIZE;
  
  int algo = UNCOUPLED;
  double epsilon = 1;
  int ssthresh = 15;
  int degree = DEFAULT_DEGREE;
  uint32_t reconf_delay = DEFAULT_RECONF_DELAY;
  string opt_method = "sipring";
  int alpha = 6;
  int no_of_nodes = DEFAULT_NODES;
  int debug = -1;

  // Default flowfile path
  string flowfile = "../../../test/taskgraph.fbuf";       // so we can read the flows from a specified file
  string weight_matrix_file = "../../../test/num_global_tokens_per_expert.txt";
  // Default log dir = ./logs/
  string logdir = "";
  string ocs_file = "";
  string ecs_file = "";
  double simtime;        // seconds
  double utiltime = .01; // seconds
  int dp_degree;
  int tp_degree;
  int pp_degree;
  int ep_degree;

  // stringstream filename(ios_base::out);
  int i = 1;
  // filename << "logout.dat";

  while (i < argc)
  {
    //   if (!strcmp(argv[i],"-o")){
    //       filename.str(std::string());
    //       filename << argv[i+1];
    //       i++;
    //   } else
    if (!strcmp(argv[i], "-nodes"))
    {
      // this is gpu num, we need to calculate the correct no_of_nodes here
      no_of_nodes = atoi(argv[i + 1]);
      cout << "no_of_nodes " << no_of_nodes << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-alpha"))
    {
      alpha = atoi(argv[i + 1]);
      cout << "alpha " << alpha << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-speed"))
    {
      SPEED = atoi(argv[i + 1]);
      cout << "speed " << SPEED << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-rtt"))
    {
      RTT = atoi(argv[i + 1]);
      cout << "RTT " << RTT << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-logdir"))
    {
        logdir = argv[i + 1];
        cout << "logdir " << logdir << endl;
        i++;
    }
    else if (!strcmp(argv[i], "-ecs_file"))
    {
        ecs_file = argv[i + 1];
        cout << "ecs_file " << argv[i + 1] << endl;
        i++;
    }
    else if (!strcmp(argv[i], "-ocs_file"))
    {
        ocs_file = argv[i + 1];
        cout << "ocs_file " << argv[i + 1] << endl;
        i++;
    }
    else if (!strcmp(argv[i], "-ssthresh"))
    {
      ssthresh = atoi(argv[i + 1]);
      cout << "ssthresh " << ssthresh << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-q"))
    {
      queuesize = memFromPkt(atoi(argv[i + 1]));
      cout << "queuesize " << queuesize << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-rdelay"))
    {
      reconf_delay = atof(argv[i + 1]);
      cout << "reconf_delay " << reconf_delay << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-dp_degree"))
    {
      dp_degree = atoi(argv[i + 1]);
      cout << "dp_degree " << dp_degree << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-tp_degree"))
    {
      tp_degree = atoi(argv[i + 1]);
      cout << "tp_degree " << tp_degree << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-pp_degree"))
    {
      pp_degree = atoi(argv[i + 1]);
      cout << "pp_degree " << pp_degree << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-ep_degree"))
    {
      ep_degree = atoi(argv[i + 1]);
      cout << "ep_degree " << ep_degree << endl;
      i++;
    }

    else if (!strcmp(argv[i], "-deg"))
    {
      degree = atoi(argv[i + 1]);
      cout << "degree " << degree << endl;
      i++;
    }
    else if (!strcmp(argv[i], "-omethod"))
    {
      opt_method = string(argv[i + 1]);
      cout << "opt_method " << opt_method << endl;
      i++;
    }
    else if (!strcmp(argv[i], "UNCOUPLED"))
      algo = UNCOUPLED;
    else if (!strcmp(argv[i], "COUPLED_INC"))
      algo = COUPLED_INC;
    else if (!strcmp(argv[i], "FULLY_COUPLED"))
      algo = FULLY_COUPLED;
    else if (!strcmp(argv[i], "COUPLED_TCP"))
      algo = COUPLED_TCP;
    else if (!strcmp(argv[i], "COUPLED_SCALABLE_TCP"))
      algo = COUPLED_SCALABLE_TCP;
    else if (!strcmp(argv[i], "COUPLED_EPSILON"))
    {
      algo = COUPLED_EPSILON;
      if (argc > i + 1)
      {
        epsilon = atof(argv[i + 1]);
        i++;
      }
      printf("Using epsilon %f\n", epsilon);
    }
    else if (!strcmp(argv[i], "-weightmatrix"))
        {
            weight_matrix_file = argv[i + 1];
            i++;
        }
    else if (!strcmp(argv[i], "-flowfile"))
    {
      flowfile = argv[i + 1];
      i++;
    }
    else if (!strcmp(argv[i], "-simtime"))
    {
      simtime = atof(argv[i + 1]);
      i++;
    }
    else if (!strcmp(argv[i], "-utiltime"))
    {
      utiltime = atof(argv[i + 1]);
      i++;
    }
    else if (!strcmp(argv[i], "-debug"))
    {
      debug = atoi(argv[i + 1]);
      FFApplication::debug_exit = debug;
      i++;
    }
    else
      exit_error(argv[0], argv[i]);
    i++;
  }
  srand(13);

  eventlist.setEndtime(timeFromSec(simtime));
  Clock c(timeFromSec(5 / 100.), eventlist);

  // if log_dir not set, use default format
    if (logdir == "")
    {
        std::string dateTime = getCurrentDateTime();
        std::string executableName = getExecutableName();
        logdir = "./logs/" + executableName + "_" + dateTime;
        std::filesystem::create_directories(logdir);

        std::cout << "Log directory created: " << logdir << std::endl;
    }
    else if (!std::filesystem::exists(logdir))
    {
        std::filesystem::create_directories(logdir);
        std::cout << "Log directory created: " << logdir << std::endl;
    }
    std::cout << "Log directory is: " << logdir << std::endl;

    string ocsfile_path;
    if (ocs_file == "") {
        ocsfile_path = logdir + "/fct_util_out.txt";
        fct_util_out_ocs.open(ocsfile_path);
    }
    else {
        // check if ofile is a path and obtain its basename
        std::filesystem::path filePath(ocs_file);
        if (filePath.has_parent_path()) {
            // 取最后一个basename作为文件名
            ocs_file = filePath.filename().string();
        }
        ocs_file = logdir + "/" + ocs_file;
        fct_util_out_ocs.open(ocs_file);
    }
    //std::cout << "Output file is: " << ofile_path << std::endl;

    string ofile_path;
    if (ecs_file == "") {
        ofile_path = logdir + "/fct_util_out.txt";
        fct_util_out_ecs.open(ofile_path);
    }
    else {
        // check if ofile is a path and obtain its basename
        std::filesystem::path filePath(ecs_file);
        if (filePath.has_parent_path()) {
            // 取最后一个basename作为文件名
            ecs_file = filePath.filename().string();
        }
        ecs_file = logdir + "/" + ecs_file;
        fct_util_out_ecs.open(ecs_file);
    }



    if (!fct_util_out_ecs.is_open()) {
        std::cerr << "Failed to open output file: " << (ecs_file == "" ? "fct_util_out.txt" : ecs_file) << std::endl;
        return 1;
    }
    if (!fct_util_out_ocs.is_open()) {
        std::cerr << "Failed to open output file: " << (ocs_file == "" ? "fct_util_out.txt" : ocs_file) << std::endl;
        return 1;
    }

  uint32_t link_speed = SPEED*(8-alpha);
  int fattree_nodenum = (no_of_nodes/8);// assume 8 NICs per node
  int fattree_node = find_nonum(fattree_nodenum);// find min k for no_of_nodes

  // Debug output for parameters
  std::cout << "\n=== Debug Parameters ===" << std::endl;
  std::cout << "SPEED: " << SPEED << std::endl;
  std::cout << "alpha: " << alpha << std::endl;
  std::cout << "link_speed: " << link_speed << std::endl;
  std::cout << "no_of_nodes: " << no_of_nodes << std::endl;
  std::cout << "fattree_nodenum: " << fattree_nodenum << std::endl;
  std::cout << "fattree_node: " << fattree_node << std::endl;
  std::cout << "queuesize: " << queuesize << std::endl;
  std::cout << "reconf_delay: " << reconf_delay << std::endl;
  std::cout << "dp_degree: " << dp_degree << std::endl;
  std::cout << "tp_degree: " << tp_degree << std::endl;
  std::cout << "pp_degree: " << pp_degree << std::endl;
  std::cout << "ep_degree: " << ep_degree << std::endl;
  std::cout << "ssthresh: " << ssthresh << std::endl;
  std::cout << "logdir: " << logdir << std::endl;
  std::cout << "========================\n" << std::endl;

  TcpRtxTimerScanner tcpRtxScanner(timeFromMs(1), eventlist);
  int max_layer_num = 100;
  int region_size = ep_degree * tp_degree / 8;
  int region_num = (no_of_nodes/8) / region_size;

  All2AllTrafficRecorder demandrecorder = All2AllTrafficRecorder(max_layer_num, region_num, region_size, &tcpRtxScanner);

  FatTreeTopology *fattree = new FatTreeTopology(fattree_node, queuesize, nullptr /*&logfile*/, &eventlist, ff, ECN, link_speed); // pass the link speed to the fat tree topology

  Mixnet *topo = new Mixnet(no_of_nodes, queuesize, nullptr /* &logfile */, eventlist, ff, ECN, 1000ULL * reconf_delay, fattree, alpha,dp_degree,tp_degree,pp_degree,ep_degree);

  MixnetTopoManager *topomanager = new MixnetTopoManager(topo, &demandrecorder, 1000ULL * reconf_delay, eventlist);
  
  FFApplication app = FFApplication(topo, ssthresh, logdir, &fct_util_out_ocs, tcpRtxScanner, eventlist, FFApplication::FF_DEFAULT_AR,1,&fct_util_out_ecs, topomanager);
  
  app.load_taskgraph_flatbuf(flowfile,weight_matrix_file);
  app.start_init_tasks();

  // UtilMonitor* UM = new UtilMonitor(top, eventlist);
  // UM->start(timeFromSec(utiltime));

  // Record the setup
  int pktsize = Packet::data_packet_size();
  // logfile.write("# pktsize=" + ntoa(pktsize) + " bytes");
  //logfile.write("# subflows=" + ntoa(subflow_count));
  // logfile.write("# hostnicrate = " + ntoa(SPEED) + " pkt/sec");
  // logfile.write("# corelinkrate = " + ntoa(SPEED*CORE_TO_HOST) + " pkt/sec");
  //logfile.write("# buffer = " + ntoa((double) (queues_na_ni[0][1]->_maxsize) / ((double) pktsize)) + " pkt");
  //double rtt = timeAsSec(timeFromUs(RTT));
  //logfile.write("# rtt =" + ntoa(rtt));

  auto start = std::chrono::high_resolution_clock::now();
  // GO!
  while (eventlist.doNextEvent())
  {
  }

  auto end = std::chrono::high_resolution_clock::now();
      // Calculate the duration
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

  // Convert the duration to hours, minutes, and seconds
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= hours;
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= minutes;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);

  // Output the duration in hour:minute:second format
  std::cout << "Total simulation duration: ";
  std::cout << std::setw(2) << std::setfill('0') << hours.count() << "h"
            << std::setw(2) << std::setfill('0') << minutes.count() << "m"
            << std::setw(2) << std::setfill('0') << seconds.count() << "s" << std::endl;

  fct_util_out_ecs << "FinalFinish " << app.final_finish_time << std::endl;
  fct_util_out_ocs << "FinalFinish " << app.final_finish_time << std::endl;
}

string ntoa(double n)
{
  stringstream s;
  s << n;
  return s.str();
}

string itoa(uint64_t n)
{
  stringstream s;
  s << n;
  return s.str();
}
