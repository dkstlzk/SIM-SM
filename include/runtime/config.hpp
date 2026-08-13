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
};

SystemConfig load_config(const std::string& filepath);

} // namespace sim_sm
