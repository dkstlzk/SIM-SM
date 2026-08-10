#pragma once

#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <fstream>
#include <iostream>

#include "architecture/warp.hpp"
#include "architecture/kernel.hpp"

namespace sim_sm {

enum class TraceLevel {
    None = 0,
    Scheduler = 1,
    Memory = 2,
    Cache = 4,
    All = 7
};

inline TraceLevel operator|(TraceLevel a, TraceLevel b) {
    return static_cast<TraceLevel>(static_cast<int>(a) | static_cast<int>(b));
}

inline TraceLevel operator&(TraceLevel a, TraceLevel b) {
    return static_cast<TraceLevel>(static_cast<int>(a) & static_cast<int>(b));
}

class TraceLogger {
public:
    TraceLogger(TraceLevel level, const std::string& output_file = "");
    ~TraceLogger();

    bool is_enabled(TraceLevel level) const;

    void log_scheduler_event(size_t sm_id, size_t cycle, const std::string& scheduler_name, size_t selected_warp_id, const std::vector<Warp>& warps);
    void log_memory_event(size_t sm_id, size_t cycle, size_t warp_id, const Instruction& inst, size_t transactions, const std::string& memory_space);
    void log_cache_event(size_t cycle, const std::string& cache_name, size_t set, size_t way, bool hit);

private:
    TraceLevel level_;
    std::unique_ptr<std::ofstream> file_stream_;
    std::ostream* out_;
};

} // namespace sim_sm
