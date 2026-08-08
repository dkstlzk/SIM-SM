#include "architecture/occupancy.hpp"
#include <algorithm>
#include <cmath>

namespace sim_sm {

OccupancyResult OccupancyCalculator::compute(const SystemConfig& config, const KernelResourceRequirements& req) {
    size_t warps_per_block = (config.block_size + config.warp_size - 1) / config.warp_size;
    size_t registers_per_block = config.block_size * req.registers_per_thread;

    size_t max_blocks_by_threads = config.max_threads_per_sm / config.block_size;

    size_t max_blocks_by_regs = (registers_per_block > 0) ?
        (config.max_registers_per_sm / registers_per_block) : config.max_blocks_per_sm;

    size_t max_blocks_by_shmem = (req.shared_memory_per_block > 0) ?
        (config.max_shared_memory_per_sm / req.shared_memory_per_block) : config.max_blocks_per_sm;

    size_t resident_blocks = std::min({
        config.max_blocks_per_sm,
        max_blocks_by_threads,
        max_blocks_by_regs,
        max_blocks_by_shmem
    });

    size_t resident_warps = resident_blocks * warps_per_block;
    size_t max_possible_warps = config.max_threads_per_sm / config.warp_size;
    double occupancy = (max_possible_warps > 0) ? static_cast<double>(resident_warps) / max_possible_warps : 0.0;

    return {resident_blocks, resident_warps, occupancy};
}

} // namespace sim_sm
