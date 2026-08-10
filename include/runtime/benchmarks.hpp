#pragma once
#include <string>

namespace sim_sm {
    void run_benchmarks(const std::string& config_path, const std::string& b_type = "all", bool debug_mode = false, int trace_level = 7, const std::string& trace_file = "");
}
