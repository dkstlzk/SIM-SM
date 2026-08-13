#include "architecture/occupancy.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sim_sm {

OccupancyResult OccupancyCalculator::compute(const SystemConfig& config, const KernelResourceRequirements& req) {
    if (config.block_size == 0 || config.warp_size == 0) {
        throw std::invalid_argument("warp_size and block_size must be > 0");
    }
    size_t warps_per_block = (config.block_size + config.warp_size - 1) / config.warp_size;
    size_t registers_per_block = config.block_size * req.registers_per_thread;

    size_t max_blocks_by_threads = config.max_threads_per_sm / config.block_size;
    size_t max_possible_warps = config.max_warps_per_sm;
    size_t max_blocks_by_warps = max_possible_warps / warps_per_block;

    size_t max_blocks_by_regs = (registers_per_block > 0) ?
        (config.max_registers_per_sm / registers_per_block) : config.max_blocks_per_sm;

    size_t max_blocks_by_shmem = (req.shared_memory_per_block > 0) ?
        (config.max_shared_memory_per_sm / req.shared_memory_per_block) : config.max_blocks_per_sm;

    size_t resident_blocks = std::min({
        config.max_blocks_per_sm,
        max_blocks_by_threads,
        max_blocks_by_warps,
        max_blocks_by_regs,
        max_blocks_by_shmem
    });

    std::string limiting_factor;
    if (resident_blocks == max_blocks_by_regs && registers_per_block > 0) {
        limiting_factor = "Registers";
    } else if (resident_blocks == max_blocks_by_shmem && req.shared_memory_per_block > 0) {
        limiting_factor = "Shared Memory";
    } else if (resident_blocks == max_blocks_by_warps) {
        limiting_factor = "Warps";
    } else if (resident_blocks == max_blocks_by_threads) {
        limiting_factor = "Threads";
    } else {
        limiting_factor = "Blocks";
    }

    size_t resident_warps = resident_blocks * warps_per_block;
    double occupancy = (max_possible_warps > 0) ? static_cast<double>(resident_warps) / max_possible_warps : 0.0;

    return {resident_blocks, resident_warps, occupancy, limiting_factor};
}

} // namespace sim_sm
