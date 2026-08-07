#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

ExecutionResult InstructionExecutor::execute(const Instruction& inst, Warp& warp, MemorySystem& memory) {
    ExecutionStatus overall_status = ExecutionStatus::Completed;
    size_t max_latency = 1; // Default latency
    size_t memory_transactions = 0;

    size_t warp_pc = warp.get_warp_pc();

    // Note: Active threads are represented implicitly by threads whose PC matches the warp_pc.
    switch (inst.opcode) {
        case Opcode::ADD: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 + val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::SUB: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 - val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::MUL: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 * val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::MOV: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                if (inst.src1 == -1) {
                    thread.registers().write(inst.dst, inst.immediate);
                } else {
                    int val = thread.registers().read(inst.src1);
                    thread.registers().write(inst.dst, val);
                }
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::LOAD: {
            std::vector<size_t> addresses;
            std::vector<size_t> active_indices;
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (static_cast<size_t>(warp.get_threads()[i].pc()) != warp_pc) continue;
                size_t base = (inst.src1 != -1) ? warp.get_threads()[i].registers().read(inst.src1) : 0;
                addresses.push_back(base + inst.immediate);
                active_indices.push_back(i);
            }

            if (!addresses.empty()) {
                std::vector<int> out_values;
                WarpMemoryResult res = memory.warp_load(addresses, out_values);
                max_latency = res.total_latency;
                memory_transactions = res.num_transactions;

                for (size_t k = 0; k < active_indices.size(); ++k) {
                    size_t thread_idx = active_indices[k];
                    warp.get_threads()[thread_idx].registers().write(inst.dst, out_values[k]);
                    warp.get_threads()[thread_idx].set_pc(warp.get_threads()[thread_idx].pc() + 1);
                }
            }
            break;
        }
        case Opcode::STORE: {
            std::vector<size_t> addresses;
            std::vector<int> values;
            std::vector<size_t> active_indices;
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (static_cast<size_t>(warp.get_threads()[i].pc()) != warp_pc) continue;
                size_t base = (inst.src2 != -1) ? warp.get_threads()[i].registers().read(inst.src2) : 0;
                addresses.push_back(base + inst.immediate);
                values.push_back(warp.get_threads()[i].registers().read(inst.src1));
                active_indices.push_back(i);
            }

            if (!addresses.empty()) {
                WarpMemoryResult res = memory.warp_store(addresses, values);
                max_latency = res.total_latency;
                memory_transactions = res.num_transactions;

                for (size_t thread_idx : active_indices) {
                    warp.get_threads()[thread_idx].set_pc(warp.get_threads()[thread_idx].pc() + 1);
                }
            }
            break;
        }
        case Opcode::CMP: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.set_predicate(val1 == val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::BRANCH: {
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) != warp_pc) continue;
                if (thread.predicate()) {
                    thread.set_pc(thread.pc() + inst.immediate);
                    overall_status = ExecutionStatus::BranchTaken;
                } else {
                    thread.set_pc(thread.pc() + 1);
                }
            }
            break;
        }
        case Opcode::BARRIER: {
            size_t active_count = 0;
            for (auto& thread : warp.get_threads()) {
                if (static_cast<size_t>(thread.pc()) == warp_pc) {
                    active_count++;
                    thread.set_pc(thread.pc() + 1);
                }
            }
            if (active_count != warp.get_threads().size()) {
                throw std::runtime_error("Divergent barrier within warp");
            }
            overall_status = ExecutionStatus::BarrierReached;
            break;
        }
        default:
            throw std::runtime_error("Unknown opcode");
    }

    return {overall_status, max_latency, memory_transactions};
}

} // namespace sim_sm
