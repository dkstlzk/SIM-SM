#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

ExecutionResult InstructionExecutor::execute(const Instruction& inst, Warp& warp, MemorySystem& memory) {
    ExecutionStatus overall_status = ExecutionStatus::Completed;
    size_t max_latency = 1; // Default latency

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
            std::vector<size_t> addresses(warp.get_threads().size(), inst.immediate);
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                int val = 0;
                size_t latency = memory.load(addresses[i], val);
                warp.get_threads()[i].registers().write(inst.dst, val);
                warp.get_threads()[i].set_pc(warp.get_threads()[i].pc() + 1);
                max_latency = std::max(max_latency, latency);
            }
            break;
        }
        case Opcode::STORE: {
            std::vector<size_t> addresses;
            std::vector<int> values;
            for (auto& thread : warp.get_threads()) {
                addresses.push_back(inst.immediate);
                values.push_back(thread.registers().read(inst.src1));
            }
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                size_t latency = memory.store(addresses[i], values[i]);
                warp.get_threads()[i].set_pc(warp.get_threads()[i].pc() + 1);
                max_latency = std::max(max_latency, latency);
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

    return {overall_status, max_latency};
}

} // namespace sim_sm
