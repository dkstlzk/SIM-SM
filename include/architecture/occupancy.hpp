#pragma once
#include "runtime/config.hpp"
#include <cstddef>
#include <string>

namespace sim_sm {

struct KernelResourceRequirements {
    size_t registers_per_thread;
    size_t shared_memory_per_block;
};

struct OccupancyResult {
    size_t resident_blocks;
    size_t resident_warps;
    double occupancy_percentage;
    std::string limiting_factor;
};

class OccupancyCalculator {
public:
    static OccupancyResult compute(const SystemConfig& config, const KernelResourceRequirements& req);
};

} // namespace sim_sm
