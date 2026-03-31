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
//#include "connection_matrix.h"

// Choose the topology here:
// #include "test_topology.h"
#include "os_fattree.h"
#include "ffapp.h"

#include <list>

// Simulation params

#define PRINT_PATHS 0

#define PERIODIC 0
#include "main.h"

uint32_t RTT_rack = 500; // us
uint32_t RTT_net = 500; // us
// int DEFAULT_NODES = 16;

uint32_t SPEED;
std::ofstream fct_util_out;

FirstFit* ff = NULL;
//unsigned int subflow_count = 8; // probably not necessary ???

#define DEFAULT_PACKET_SIZE 1500 // full packet (including header), Bytes
#define DEFAULT_HEADER_SIZE 64 // header size, Bytes
#define DEFAULT_QUEUE_SIZE 200

#define DEFAULT_SPEED 40000

string ntoa(double n);
string itoa(uint64_t n);

EventList eventlist;
// Logfile* lg;

void exit_error(char* progr, char *param) {
    cerr << "Bad parameter: " << param << endl;
    cerr << "Usage " << progr << " [UNCOUPLED(DEFAULT)|COUPLED_INC|FULLY_COUPLED|COUPLED_EPSILON] [epsilon][COUPLED_SCALABLE_TCP]" << endl;
    exit(1);
}

void print_path(std::ofstream &paths, const Route* rt){
    for (unsigned int i=1;i<rt->size()-1;i+=2){
  RandomQueue* q = (RandomQueue*)rt->at(i);
  if (q!=NULL)
      paths << q->str() << " ";
  else 
      paths << "NULL ";
    }

    paths<<endl;
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

int main(int argc, char **argv) {

    TcpPacket::set_packet_size(DEFAULT_PACKET_SIZE - DEFAULT_HEADER_SIZE); // MTU
    mem_b queuesize = DEFAULT_QUEUE_SIZE * DEFAULT_PACKET_SIZE;

    int algo = UNCOUPLED;
    double epsilon = 1;
    int ssthresh = 15;

    int k;
    int nlp;

    // Default flowfile path
    string flowfile = "../../../test/taskgraph.fbuf";       // so we can read the flows from a specified file
    string weight_matrix_file = "../../../test/num_global_tokens_per_expert.txt";
    // Default log dir = ./logs/
    string logdir = "";
    string ofile = "";

    double simtime; // seconds
    double utiltime=.01; // seconds

    // stringstream filename(ios_base::out);
    int i = 1;
    // filename << "logout.dat";

    while (i<argc) {
//   if (!strcmp(argv[i],"-o")){
//       filename.str(std::string());
//       filename << argv[i+1];
//       i++;
//   } else 
  if (!strcmp(argv[i],"-k")){
      k = atoi(argv[i+1]);
      cout << "K "<<k << endl;
      i++;
  } else if (!strcmp(argv[i],"-nlp")){
      nlp = atoi(argv[i+1]);
      cout << "nlp "<<nlp << endl;
      i++;
  } else if (!strcmp(argv[i],"-speed")){
      SPEED = atoi(argv[i+1]);
      cout << "speed "<<SPEED << endl;
      i++;
  } else if (!strcmp(argv[i],"-rttrack")){
      RTT_rack = atoi(argv[i+1]);
      cout << "RTT_rack "<<RTT_rack << endl;
      i++;
  } else if (!strcmp(argv[i],"-rttnet")){
      RTT_net = atoi(argv[i+1]);
      cout << "rttnet "<<RTT_net << endl;
      i++;
  }  else if (!strcmp(argv[i], "-logdir"))
  {
      logdir = argv[i + 1];
      cout << "logdir " << logdir << endl;
      i++;
  }
  else if (!strcmp(argv[i], "-weightmatrix"))
        {
            weight_matrix_file = argv[i + 1];
            i++;
        }
  else if (!strcmp(argv[i], "-ofile"))
  {
      ofile = argv[i + 1];
      cout << "ofile " << argv[i + 1] << endl;
      i++;
  } else if (!strcmp(argv[i],"-ssthresh")){
      ssthresh = atoi(argv[i+1]);
      cout << "ssthresh "<< ssthresh << endl;
      i++;
  } else if (!strcmp(argv[i],"-q")){
      queuesize = memFromPkt(atoi(argv[i+1]));
      cout << "queuesize "<<queuesize << endl;
      i++;
  } else if (!strcmp(argv[i], "UNCOUPLED"))
      algo = UNCOUPLED;
  else if (!strcmp(argv[i], "COUPLED_INC"))
      algo = COUPLED_INC;
  else if (!strcmp(argv[i], "FULLY_COUPLED"))
      algo = FULLY_COUPLED;
  else if (!strcmp(argv[i], "COUPLED_TCP"))
      algo = COUPLED_TCP;
  else if (!strcmp(argv[i], "COUPLED_SCALABLE_TCP"))
      algo = COUPLED_SCALABLE_TCP;
  else if (!strcmp(argv[i], "COUPLED_EPSILON")) {
      algo = COUPLED_EPSILON;
      if (argc > i+1){
    epsilon = atof(argv[i+1]);
    i++;
      }
      printf("Using epsilon %f\n", epsilon);
  } else if (!strcmp(argv[i],"-flowfile")) {
    flowfile = argv[i+1];
    i++;
    } else if (!strcmp(argv[i],"-simtime")) {
        simtime = atof(argv[i+1]);
        i++;
  } else if (!strcmp(argv[i],"-utiltime")) {
        utiltime = atof(argv[i+1]);
        i++;
  } else
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

    string ofile_path;
    if (ofile == "") {
        ofile_path = logdir + "/fct_util_out.txt";
        fct_util_out.open(ofile_path);
    }
    else {
        // check if ofile is a path and obtain its basename
        std::filesystem::path filePath(ofile);
        if (filePath.has_parent_path()) {
            // 取最后一个basename作为文件名
            ofile = filePath.filename().string();
        }
        ofile_path = logdir + "/" + ofile;
        fct_util_out.open(ofile_path);
    }
    std::cout << "Output file is: " << ofile_path << std::endl;

    if (!fct_util_out.is_open()) {
        std::cerr << "Failed to open output file: " << (ofile == "" ? "fct_util_out.txt" : ofile) << std::endl;
        return 1;
    }

    std::cout << "Output file is open and ready for writing: " << (ofile == "" ? "fct_util_out.txt" : ofile) << std::endl;

      
    //cout <<  "Using algo="<<algo<< " epsilon=" << epsilon << endl;

    // Logfile logfile(filename.str(), eventlist);

#if PRINT_PATHS
    filename << ".paths";
    cout << "Logging path choices to " << filename.str() << endl;
    std::ofstream paths(filename.str().c_str());
    if (!paths){
  cout << "Can't open for writing paths file!"<<endl;
  exit(1);
    }
#endif

    // lg = &logfile;

    


    // !!!!!!!!!!!!!!!!!!!!!!!
    // logfile.setStartTime(timeFromSec(10));






    // TcpSinkLoggerSampling sinkLogger = TcpSinkLoggerSampling(timeFromUs(50.), eventlist);
    // logfile.addLogger(sinkLogger);
    // TcpTrafficLogger traffic_logger = TcpTrafficLogger();
    // traffic_logger.fct_util_out = &fct_util_out;
    // logfile.addLogger(traffic_logger);

    TcpRtxTimerScanner tcpRtxScanner(timeFromMs(1), eventlist);

    OverSubscribedFatTree* top = new OverSubscribedFatTree(k, nlp, queuesize, nullptr /* &logfile */, &eventlist, ff, ECN);

    // FFApplication app = FFApplication(top, ssthresh, sinkLogger, traffic_logger, tcpRtxScanner, eventlist);
    FFApplication app = FFApplication(top, ssthresh, logdir, &fct_util_out, tcpRtxScanner, eventlist);
    //app.load_taskgraph_flatbuf(flowfile);
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
    while (eventlist.doNextEvent()) {
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

    fct_util_out << "Final finish time: " << app.final_finish_time << std::endl;  // Changed to match aggosft format
}

string ntoa(double n) {
    stringstream s;
    s << n;
    return s.str();
}

string itoa(uint64_t n) {
    stringstream s;
    s << n;
    return s.str();
}
