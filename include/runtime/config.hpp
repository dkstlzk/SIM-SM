#pragma once

#include <string>
#include <cstddef>

namespace sim_sm {

struct SystemConfig {
    size_t num_sms;
    size_t warp_size;
    size_t block_size;
    
    // Occupancy limits
    size_t max_threads_per_sm;
    size_t max_blocks_per_sm;
    size_t max_shared_memory_per_sm;
    size_t max_registers_per_sm;
    size_t max_warps_per_sm = 64;

    // Cache hierarchy
    size_t l1_sets = 4;
    size_t l1_associativity = 4;
    size_t l1_line_size = 32;
    std::string l1_policy = "LRU";

    size_t l2_sets = 16;
    size_t l2_associativity = 8;
    size_t l2_line_size = 32;
    std::string l2_policy = "LRU";

    // Latencies
    size_t shared_memory_latency = 1;
    size_t l1_latency = 5;
    size_t l2_latency = 20;
    size_t global_memory_latency = 100;
    size_t writeback_latency = 20;

    // Shared memory banking
    size_t shared_memory_banks = 32;

    // Global memory size
    size_t global_memory_size = 1048576;
};

SystemConfig load_config(const std::string& filepath);

} // namespace sim_sm
