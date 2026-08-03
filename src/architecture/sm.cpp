#include "architecture/sm.hpp"
#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

SM::SM(size_t sm_id) : sm_id_(sm_id) {}

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

void SM::tick(const Kernel& kernel, FlatMemory& memory) {
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
            counters_.add_stall(StallReason::SyntheticLatency);
        }
        return;
    }

    // 3. Fetch instruction
    size_t pc = selected_warp->get_warp_pc();
    if (pc >= kernel.instructions().size()) {
        throw std::runtime_error("Invariant violation: non-completed warp has invalid PC.");
    }
    const Instruction& inst = kernel.instructions()[pc];

    // 4. Execute for all threads (enforcing no divergence)
    for (auto& thread : selected_warp->get_threads()) {
        if (static_cast<size_t>(thread.pc()) != pc) {
            throw std::runtime_error("Invariant violation: thread PC diverged from warp PC prior to execution.");
        }
        InstructionExecutor::execute(inst, thread, memory);
    }

    // 5. Check and update PC
    size_t next_pc = pc + 1;
    if (!selected_warp->get_threads().empty()) {
        next_pc = static_cast<size_t>(selected_warp->get_threads()[0].pc());
        for (const auto& thread : selected_warp->get_threads()) {
            if (static_cast<size_t>(thread.pc()) != next_pc) {
                throw std::runtime_error("Invariant violation: thread PCs diverged after execution.");
            }
        }
    }
    selected_warp->set_warp_pc(next_pc);

    // 6. State transition and synthetic latency
    if (next_pc >= kernel.instructions().size()) {
        selected_warp->set_completed();
    } else if (inst.opcode == Opcode::LOAD || inst.opcode == Opcode::STORE || inst.opcode == Opcode::MUL) {
        selected_warp->stall(5);
    }

    counters_.increment_instructions_retired();
}

} // namespace sim_sm
