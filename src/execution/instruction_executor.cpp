#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

ExecutionResult InstructionExecutor::execute(const Instruction& inst, Warp& warp, MemorySystem& memory) {
    ExecutionStatus overall_status = ExecutionStatus::Completed;
    size_t max_latency = 1; // Default latency
    size_t memory_transactions = 0;

    switch (inst.opcode) {
        case Opcode::ADD: {
            for (auto& thread : warp.get_threads()) {
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 + val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::SUB: {
            for (auto& thread : warp.get_threads()) {
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 - val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::MUL: {
            for (auto& thread : warp.get_threads()) {
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 * val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::MOV: {
            for (auto& thread : warp.get_threads()) {
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
            std::vector<size_t> addresses(warp.get_threads().size());
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                size_t base = (inst.src1 != -1) ? warp.get_threads()[i].registers().read(inst.src1) : 0;
                addresses[i] = base + inst.immediate;
            }

            std::vector<int> out_values;
            WarpMemoryResult res = memory.warp_load(addresses, out_values);
            max_latency = res.total_latency;
            memory_transactions = res.num_transactions;

            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                warp.get_threads()[i].registers().write(inst.dst, out_values[i]);
                warp.get_threads()[i].set_pc(warp.get_threads()[i].pc() + 1);
            }
            break;
        }
        case Opcode::STORE: {
            std::vector<size_t> addresses(warp.get_threads().size());
            std::vector<int> values(warp.get_threads().size());
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                // src2 is used as base address register for store if provided, otherwise 0
                size_t base = (inst.src2 != -1) ? warp.get_threads()[i].registers().read(inst.src2) : 0;
                addresses[i] = base + inst.immediate;
                values[i] = warp.get_threads()[i].registers().read(inst.src1);
            }

            WarpMemoryResult res = memory.warp_store(addresses, values);
            max_latency = res.total_latency;
            memory_transactions = res.num_transactions;

            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                warp.get_threads()[i].set_pc(warp.get_threads()[i].pc() + 1);
            }
            break;
        }
        case Opcode::CMP: {
            for (auto& thread : warp.get_threads()) {
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.set_predicate(val1 == val2);
                thread.set_pc(thread.pc() + 1);
            }
            break;
        }
        case Opcode::BRANCH: {
            for (auto& thread : warp.get_threads()) {
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
            for (auto& thread : warp.get_threads()) {
                thread.set_pc(thread.pc() + 1);
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
