#include "architecture/sm.hpp"
#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

SM::SM(size_t sm_id, size_t l1_sets, size_t l1_assoc, size_t l1_line_size, const std::string& cache_policy)
    : sm_id_(sm_id), l1_cache_(l1_sets, l1_assoc, l1_line_size, cache_policy), shared_memory_(65536) {}

size_t SM::get_sm_id() const {
    return sm_id_;
}

void SM::add_warp(const Warp& warp) {
    warps_.push_back(warp);
}

void SM::set_scheduler(std::unique_ptr<WarpScheduler> scheduler) {
    scheduler_ = std::move(scheduler);
}

const PerformanceCounter& SM::get_counters() const {
    return counters_;
}

std::vector<Warp>& SM::get_warps() {
    return warps_;
}

bool SM::is_completed() const {
    if (warps_.empty()) return true;
    for (const auto& warp : warps_) {
        if (warp.get_state() != WarpState::Completed) {
            return false;
        }
    }
    return true;
}

void SM::clear_warps() {
    warps_.clear();
}

void SM::tick(const Kernel& kernel, MemorySystem& memory) {
    if (is_completed()) {
        return;
    }

    // 1. Tick stalled warps
    for (auto& warp : warps_) {
        warp.tick_stall();
    }

    counters_.increment_cycles();

    // 2. Select warp
    if (!scheduler_) {
        counters_.add_stall(StallReason::NoReadyWarp);
        return;
    }

    Warp* selected_warp = scheduler_->select_warp(warps_);
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

    if (inst.opcode == Opcode::LOAD || inst.opcode == Opcode::STORE) {
        counters_.increment_memory_instructions();
        counters_.add_memory_transactions(result.memory_transactions);
    }

    if (result.status == ExecutionStatus::BarrierReached) {
        handle_barrier_arrival(*selected_warp);
    }

    // 5. Update PC - Minimum PC of all non-completed/active threads
    // Note: Active threads are represented implicitly by threads whose PC remains within the kernel instruction stream.
    size_t min_pc = SIZE_MAX;
    for (const auto& thread : selected_warp->get_threads()) {
        if (thread.pc() < 0) {
            throw std::runtime_error("Invariant violation: thread PC is negative.");
        }
        size_t tpc = static_cast<size_t>(thread.pc());
        if (tpc < kernel.instructions().size()) {
            if (tpc < min_pc) {
                min_pc = tpc;
            }
        } else if (tpc > kernel.instructions().size()) {
            throw std::runtime_error("Invariant violation: thread PC jumped beyond kernel bounds.");
        }
    }

    if (min_pc == SIZE_MAX) {
        selected_warp->set_warp_pc(kernel.instructions().size());
        if (selected_warp->get_state() != WarpState::StalledAtBarrier) {
            selected_warp->set_completed();
        }
    } else {
        selected_warp->set_warp_pc(min_pc);
        if (result.status != ExecutionStatus::BarrierReached && result.latency > 1) {
            selected_warp->stall(result.latency - 1);
        }
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
