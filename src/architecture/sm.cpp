#include "architecture/sm.hpp"
#include "execution/instruction_executor.hpp"
#include "runtime/trace_logger.hpp"
#include <stdexcept>
#include <algorithm>

namespace sim_sm {

SM::SM(size_t sm_id, const SystemConfig& config)
    : sm_id_(sm_id), l1_cache_(config.l1_sets, config.l1_associativity, config.l1_line_size, config.l1_policy), shared_memory_(config.max_shared_memory_per_sm, config.shared_memory_banks) {}

SM::SM(size_t sm_id, size_t l1_sets, size_t l1_assoc, size_t l1_line_size, const std::string& cache_policy)
    : sm_id_(sm_id), l1_cache_(l1_sets, l1_assoc, l1_line_size, cache_policy), shared_memory_(65536, 32) {}

size_t SM::get_sm_id() const {
    return sm_id_;
}

void SM::add_warp(const Warp& warp) {
    warps_.push_back(warp);
}

void SM::set_scheduler(std::unique_ptr<WarpScheduler> scheduler) {
    scheduler_ = std::move(scheduler);
}

void SM::set_trace_logger(TraceLogger* logger) {
    counters_.set_trace_logger(logger);
    l1_cache_.set_event_callback([logger, this](const CacheEvent& e) {
        if (logger) {
            logger->log_cache_event(counters_.get_cycles(), "L1$" + std::to_string(sm_id_), e.set, e.way, e.hit);
        }
    });
}

const PerformanceCounter& SM::get_counters() const {
    return counters_;
}

std::vector<Warp>& SM::get_warps() {
    return warps_;
}

bool SM::is_completed() const {
    if (warps_.empty() && resident_blocks_.empty()) return true;
    for (const auto& warp : warps_) {
        if (warp.get_state() != WarpState::Completed) {
            return false;
        }
    }
    return true;
}

void SM::clear_warps() {
    warps_.clear();
    resident_blocks_.clear();
    allocated_blocks_ = 0;
    allocated_warps_ = 0;
    allocated_threads_ = 0;
    allocated_registers_ = 0;
    allocated_shared_memory_ = 0;
}

bool SM::can_admit(const ThreadBlock& block, const SystemConfig& config, const KernelResourceRequirements& req) const {
    if (allocated_blocks_ + 1 > config.max_blocks_per_sm) return false;
    
    size_t block_warps = block.get_warps().size();
    size_t block_threads = 0;
    for (const auto& w : block.get_warps()) {
        block_threads += w.get_threads().size();
    }
    
    size_t regs_needed = block_threads * req.registers_per_thread;
    size_t shmem_needed = req.shared_memory_per_block;
    
    size_t max_warps_per_sm = config.max_warps_per_sm;
    if (allocated_warps_ + block_warps > max_warps_per_sm) return false;
    if (allocated_threads_ + block_threads > config.max_threads_per_sm) return false;
    if (allocated_registers_ + regs_needed > config.max_registers_per_sm) return false;
    if (allocated_shared_memory_ + shmem_needed > config.max_shared_memory_per_sm) return false;
    
    return true;
}

void SM::allocate_block(const ThreadBlock& block, const SystemConfig& config, const KernelResourceRequirements& req) {
    (void)config;
    size_t block_warps = block.get_warps().size();
    size_t block_threads = 0;
    for (const auto& w : block.get_warps()) {
        block_threads += w.get_threads().size();
    }
    size_t regs_needed = block_threads * req.registers_per_thread;
    size_t shmem_needed = req.shared_memory_per_block;
    
    allocated_blocks_++;
    allocated_warps_ += block_warps;
    allocated_threads_ += block_threads;
    allocated_registers_ += regs_needed;
    allocated_shared_memory_ += shmem_needed;
    
    resident_blocks_.push_back({block.get_block_id(), block_threads, block_warps, regs_needed, shmem_needed});
    
    for (const auto& warp : block.get_warps()) {
        add_warp(warp);
    }
}

void SM::release_completed_blocks() {
    auto block_it = resident_blocks_.begin();
    while (block_it != resident_blocks_.end()) {
        bool all_completed = true;
        for (const auto& w : warps_) {
            if (!w.get_threads().empty() && w.get_threads()[0].get_block_id() == block_it->block_id) {
                if (w.get_state() != WarpState::Completed) {
                    all_completed = false;
                    break;
                }
            }
        }
        
        if (all_completed) {
            allocated_blocks_--;
            allocated_warps_ -= block_it->num_warps;
            allocated_threads_ -= block_it->num_threads;
            allocated_registers_ -= block_it->allocated_registers;
            allocated_shared_memory_ -= block_it->allocated_shared_memory;
            
            warps_.erase(std::remove_if(warps_.begin(), warps_.end(),
                [&](const Warp& w) {
                    return !w.get_threads().empty() && w.get_threads()[0].get_block_id() == block_it->block_id;
                }), warps_.end());
            
            block_it = resident_blocks_.erase(block_it);
        } else {
            ++block_it;
        }
    }
}

void SM::tick(const Kernel& kernel, MemorySystem& memory) {
    if (is_completed()) {
        return;
    }

    // 1. Tick stalled warps
    size_t barrier_stalls = 0;
    for (auto& warp : warps_) {
        warp.tick_stall();
        if (warp.get_state() == WarpState::StalledAtBarrier) {
            barrier_stalls++;
        }
    }
    
    if (barrier_stalls > 0) {
        counters_.add_warp_barrier_stall_cycles(barrier_stalls);
    }

    counters_.increment_cycles();

    // 2. Select warp
    if (!scheduler_) {
        counters_.add_stall(StallReason::NoReadyWarp);
        return;
    }

    Warp* selected_warp = scheduler_->select_warp(warps_);
    
    counters_.record_scheduler_event(sm_id_, counters_.get_cycles(), scheduler_->name(), selected_warp ? selected_warp->get_warp_id() : -1, warps_);

    if (!selected_warp) {
        // Differentiate stall reason: if warps exist but none ready, they are likely stalled on synthetic latency.
        if (warps_.empty()) {
            counters_.add_stall(StallReason::NoReadyWarp);
        } else {
            // Check for potential deadlock if all active warps are stalled
            // In our current constrained model, true silent deadlocks (where all warps are indefinitely stalled without being malformed)
            // are structurally unreachable because legal barriers eventually release and malformed barriers throw immediately.
        }
        return;
    }

    // 3. Fetch instruction
    size_t pc = selected_warp->get_warp_pc();
    if (pc >= kernel.instructions().size()) {
        selected_warp->set_completed();
        return;
    }
    const Instruction& inst = kernel.instructions()[pc];

    // 4. Execute for active threads
    ExecutionResult result = InstructionExecutor::execute(inst, *selected_warp, memory);

    if (inst.opcode == Opcode::LOAD || inst.opcode == Opcode::STORE || inst.opcode == Opcode::ATOMIC_ADD) {
        counters_.increment_memory_instructions();
        counters_.add_memory_transactions(result.memory_transactions);
        counters_.record_memory_event(sm_id_, counters_.get_cycles(), selected_warp->get_warp_id(), inst, result.memory_transactions, result.memory_space);
    }

    if (result.bank_conflicts > 0) {
        counters_.add_bank_conflict_stalls(result.bank_conflicts);
    }
    if (result.dirty_evictions > 0) {
        counters_.add_dirty_eviction_writebacks(result.dirty_evictions);
    }

    if (result.status == ExecutionStatus::BarrierReached) {
        handle_barrier_arrival(*selected_warp);
    }

    // 5. Update Warp State
    if (selected_warp->get_warp_pc() >= kernel.instructions().size()) {
        SIMTStackEntry next_path;
        if (selected_warp->pop_simt_stack(next_path)) {
            selected_warp->set_active_mask(next_path.active_mask);
            selected_warp->set_warp_pc(next_path.target_pc);
            selected_warp->set_reconvergence_pc(next_path.reconvergence_pc);
            
            // Update shadow PCs for the popped path
            for (size_t i = 0; i < selected_warp->get_threads().size(); ++i) {
                if (next_path.active_mask.test(i)) {
                    selected_warp->get_threads()[i].set_pc(next_path.target_pc);
                }
            }
        } else {
            if (selected_warp->get_state() != WarpState::StalledAtBarrier) {
                selected_warp->set_completed();
            }
        }
    } else {
        if (result.status != ExecutionStatus::BarrierReached && result.latency > 1) {
            selected_warp->stall(result.latency - 1);
        }
    }

    for (size_t i = 0; i < result.write_conflict_stalls; ++i) {
        counters_.add_stall(StallReason::WriteConflict);
    }

    counters_.increment_instructions_retired();
}

void SM::handle_barrier_arrival(Warp& warp) {
    warp.set_stalled_at_barrier();

    // Note: Block barriers assume all warps belonging to a resident block execute on the same SM,
    // consistent with the current block-level dispatch model.
    size_t block_id = warp.get_threads().empty() ? 0 : warp.get_threads()[0].get_block_id();

    size_t total_warps = 0;
    size_t completed_warps = 0;
    size_t stalled_warps = 0;

    for (const auto& w : warps_) {
        if (!w.get_threads().empty() && w.get_threads()[0].get_block_id() == block_id) {
            total_warps++;
            if (w.get_state() == WarpState::Completed) completed_warps++;
            if (w.get_state() == WarpState::StalledAtBarrier) stalled_warps++;
        }
    }

    if (stalled_warps + completed_warps == total_warps) {
        if (completed_warps > 0) {
            throw std::runtime_error("Malformed barrier: some warps in the block completed without reaching the barrier.");
        }
        // All warps arrived, release them
        for (auto& w : warps_) {
            if (!w.get_threads().empty() && w.get_threads()[0].get_block_id() == block_id) {
                if (w.get_state() == WarpState::StalledAtBarrier) {
                    w.set_ready();
                }
            }
        }
    }
}

} // namespace sim_sm
