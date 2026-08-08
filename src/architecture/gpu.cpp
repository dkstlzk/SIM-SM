#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/kernel.hpp"
#include "architecture/occupancy.hpp"
#include "memory/memory_system.hpp"
#include <iostream>

namespace sim_sm {

GPU::GPU(size_t num_sms, size_t l1_sets, size_t l1_assoc, size_t l2_sets, size_t l2_assoc, size_t l2_line_size, size_t global_mem_size)
    : l2_cache_(l2_sets, l2_assoc, l2_line_size, "LRU"), global_memory_(global_mem_size) {
    for (size_t i = 0; i < num_sms; ++i) {
        sms_.emplace_back(i, l1_sets, l1_assoc, 32);
    }
}

void GPU::launch_kernel(const Kernel& kernel, const Grid& grid, const SystemConfig& config, const KernelResourceRequirements& req) {
    OccupancyResult occ = OccupancyCalculator::compute(config, req);
    resident_blocks_per_sm_ = occ.resident_blocks;
    pending_blocks_ = grid.get_blocks();
    current_block_idx_ = 0;

    // Initial dispatch
    for (auto& sm : sms_) {
        sm.clear_warps();
        size_t blocks_added = 0;
        while (blocks_added < resident_blocks_per_sm_ && current_block_idx_ < pending_blocks_.size()) {
            const auto& block = pending_blocks_[current_block_idx_];
            for (const auto& warp : block.get_warps()) {
                sm.add_warp(warp);
            }
            blocks_added++;
            current_block_idx_++;
        }
    }
}

void GPU::run_to_completion(const Kernel& kernel) {
    bool all_done = false;
    while (!all_done) {
        all_done = true;

        // Phase 1: Load new blocks into empty SMs
        for (auto& sm : sms_) {
            if (sm.is_completed()) {
                if (current_block_idx_ < pending_blocks_.size()) {
                    sm.clear_warps();
                    size_t blocks_added = 0;
                    while (blocks_added < resident_blocks_per_sm_ && current_block_idx_ < pending_blocks_.size()) {
                        const auto& block = pending_blocks_[current_block_idx_];
                        for (const auto& warp : block.get_warps()) {
                            sm.add_warp(warp);
                        }
                        blocks_added++;
                        current_block_idx_++;
                    }
                }
            }
        }

        // Phase 2: Tick active SMs
        for (auto& sm : sms_) {
            if (!sm.is_completed()) {
                MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), l2_cache_, global_memory_);
                sm.tick(kernel, mem);
                all_done = false;
            }
        }
    }
}

std::vector<SM>& GPU::get_sms() {
    return sms_;
}

const std::vector<SM>& GPU::get_sms() const {
    return sms_;
}

} // namespace sim_sm
