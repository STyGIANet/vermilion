#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "taskgraph_generated.h"

using namespace FlatBufTaskGraph;

namespace {
std::string escape_json(const std::string &in) {
    std::ostringstream oss;
    for (char c : in) {
        switch (c) {
        case '\"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                oss << "\\u"
                    << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(c));
            } else {
                oss << c;
            }
        }
    }
    return oss.str();
}

template <typename Vec>
void print_uint64_array(std::ostream &out, const Vec *vec) {
    if (!vec) {
        out << "[]";
        return;
    }
    out << "[";
    for (flatbuffers::uoffset_t i = 0; i < vec->size(); i++) {
        if (i)
            out << ",";
        out << vec->Get(i);
    }
    out << "]";
}

void dump_taskgraph(const TaskGraph *tg, std::ostream &out) {
    out << "{";
    out << "\"ngpupernode\":" << tg->ngpupernode() << ",";
    out << "\"nnode\":" << tg->nnode() << ",";
    out << "\"intergpubw\":" << tg->intergpubw() << ",";
    out << "\"drambw\":" << tg->drambw() << ",";
    out << "\"netbw\":" << tg->netbw() << ",";
    out << "\"dp_degree\":" << tg->dp_degree() << ",";
    out << "\"tp_degree\":" << tg->tp_degree() << ",";
    out << "\"pp_degree\":" << tg->pp_degree() << ",";
    out << "\"ep_degree\":" << tg->ep_degree() << ",";

    // Devices
    out << "\"devices\":[";
    const auto *devices = tg->devices();
    if (devices) {
        for (flatbuffers::uoffset_t i = 0; i < devices->size(); i++) {
            const auto *d = devices->Get(i);
            if (i)
                out << ",";
            out << "{";
            out << "\"type\":\"" << EnumNameDeviceType(d->type()) << "\",";
            out << "\"deviceid\":" << d->deviceid() << ",";
            out << "\"nodeid\":" << d->nodeid() << ",";
            out << "\"deviceproperty\":" << d->deviceproperty() << ",";
            out << "\"bandwidth\":" << d->bandwidth();
            out << "}";
        }
    }
    out << "],";

    // Operators
    out << "\"ops\":[";
    const auto *ops = tg->ops();
    if (ops) {
        for (flatbuffers::uoffset_t i = 0; i < ops->size(); i++) {
            const auto *op = ops->Get(i);
            if (i)
                out << ",";
            out << "{";
            out << "\"opid\":" << op->opid() << ",";
            out << "\"type\":" << op->type() << ",";
            out << "\"name\":\"" << (op->name() ? escape_json(op->name()->str()) : "") << "\"";
            out << "}";
        }
    }
    out << "],";

    // Tasks
    out << "\"tasks\":[";
    const auto *tasks = tg->tasks();
    if (tasks) {
        for (flatbuffers::uoffset_t i = 0; i < tasks->size(); i++) {
            const auto *t = tasks->Get(i);
            if (i)
                out << ",";
            out << "{";
            out << "\"type\":\"" << EnumNameSimTaskType(t->type()) << "\",";
            out << "\"taskid\":" << t->taskid() << ",";
            out << "\"deviceid\":" << t->deviceid() << ",";
            out << "\"runtime\":" << t->runtime() << ",";
            out << "\"xfersize\":" << t->xfersize() << ",";
            out << "\"counter\":" << t->counter() << ",";
            out << "\"nexttasks\":";
            print_uint64_array(out, t->nexttasks());
            out << ",\"from_node_ids\":";
            print_uint64_array(out, t->from_node_ids());
            out << ",\"to_node_ids\":";
            print_uint64_array(out, t->to_node_ids());
            out << ",\"name\":\"" << (t->name() ? escape_json(t->name()->str()) : "") << "\",";
            out << "\"info\":\"" << (t->info() ? escape_json(t->info()->str()) : "") << "\",";
            out << "\"micro_batch_id\":" << t->micro_batch_id() << ",";
            out << "\"layer_id\":" << t->layer_id() << ",";
            out << "\"target_micro_batch_id\":" << t->target_micro_batch_id() << ",";
            out << "\"target_layer_id\":" << t->target_layer_id();
            out << "}";
        }
    }
    out << "]";

    out << "}";
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.fbuf> <output.json>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    std::ifstream infile(input_path, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "Failed to open input: " << input_path << "\n";
        return 1;
    }
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!infile.read(buffer.data(), size)) {
        std::cerr << "Failed to read input file\n";
        return 1;
    }

    const TaskGraph *tg = nullptr;
    {
        flatbuffers::Verifier::Options opts;
        opts.max_tables = 10000000; // allow large graphs
        opts.max_depth = 64;
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(buffer.data()), buffer.size(), opts);
        if (verifier.VerifyBuffer<TaskGraph>(nullptr)) {
            tg = flatbuffers::GetRoot<TaskGraph>(buffer.data());
        } else if (verifier.VerifySizePrefixedBuffer<TaskGraph>(nullptr)) {
            tg = flatbuffers::GetSizePrefixedRoot<TaskGraph>(buffer.data());
        } else {
            std::cerr << "FlatBuffer verification failed for: " << input_path << "\n";
            return 1;
        }
    }

    std::ofstream outfile(output_path);
    if (!outfile) {
        std::cerr << "Failed to open output: " << output_path << "\n";
        return 1;
    }
    dump_taskgraph(tg, outfile);
    std::cerr << "Wrote JSON to " << output_path << "\n";
    return 0;
}
