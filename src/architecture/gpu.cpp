#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/kernel.hpp"
#include "architecture/occupancy.hpp"
#include "memory/memory_system.hpp"
#include "runtime/trace_logger.hpp"
#include <iostream>

namespace sim_sm {

GPU::GPU(const SystemConfig& config)
    : config_(config), l2_cache_(config.l2_sets, config.l2_associativity, config.l2_line_size, config.l2_policy), global_memory_(config.global_memory_size) {
    for (size_t i = 0; i < config.num_sms; ++i) {
        sms_.emplace_back(i, config);
    }
}

GPU::GPU(size_t num_sms, size_t l1_sets, size_t l1_assoc, size_t l2_sets, size_t l2_assoc, size_t l2_line_size, size_t global_mem_size, const std::string& cache_policy)
    : l2_cache_(l2_sets, l2_assoc, l2_line_size, cache_policy), global_memory_(global_mem_size) {
    for (size_t i = 0; i < num_sms; ++i) {
        sms_.emplace_back(i, l1_sets, l1_assoc, 32, cache_policy);
    }
}

void GPU::launch_kernel(const Kernel& kernel, const Grid& grid, const SystemConfig& config, const KernelResourceRequirements& req) {
    (void)kernel;
    config_ = config;
    req_ = req;
    pending_blocks_ = grid.get_blocks();
    current_block_idx_ = 0;

    // Initial dispatch
    for (auto& sm : sms_) {
        sm.clear_warps();
        while (current_block_idx_ < pending_blocks_.size()) {
            const auto& block = pending_blocks_[current_block_idx_];
            if (sm.can_admit(block, config_, req_)) {
                sm.allocate_block(block, config_, req_);
                current_block_idx_++;
            } else {
                if (sm.get_allocated_blocks() == 0) {
                    throw std::runtime_error("Architecturally impossible block: exceeds empty SM limits.");
                }
                break; // SM is full
            }
        }
    }
}

void GPU::run_to_completion(const Kernel& kernel) {
    bool all_done = false;
    while (!all_done) {
        all_done = true;

        // Phase 1: Try to load new blocks into SMs that have available resources
        for (auto& sm : sms_) {
            sm.release_completed_blocks();
            while (current_block_idx_ < pending_blocks_.size()) {
                const auto& block = pending_blocks_[current_block_idx_];
                if (sm.can_admit(block, config_, req_)) {
                    sm.allocate_block(block, config_, req_);
                    current_block_idx_++;
                } else {
                    if (sm.get_allocated_blocks() == 0) {
                        throw std::runtime_error("Architecturally impossible block: exceeds empty SM limits.");
                    }
                    break;
                }
            }
        }

        // Phase 2: Tick active SMs
        for (auto& sm : sms_) {
            if (!sm.is_completed()) {
                MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), l2_cache_, global_memory_, MemoryAccessConfig{
                    config_.shared_memory_latency,
                    config_.l1_latency,
                    config_.l2_latency,
                    config_.global_memory_latency,
                    config_.writeback_latency
                });
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

void GPU::set_trace_logger(TraceLogger* logger) {
    for (auto& sm : sms_) {
        sm.set_trace_logger(logger);
    }
    l2_cache_.set_event_callback([logger](const CacheEvent& e) {
        if (logger) {
            logger->log_cache_event(e.cycle, "L2$", e.set, e.way, e.hit);
        }
    });
}

} // namespace sim_sm
