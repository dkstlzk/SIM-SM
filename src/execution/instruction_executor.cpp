#include "execution/instruction_executor.hpp"
#include <stdexcept>

namespace sim_sm {

ExecutionStatus InstructionExecutor::execute(const Instruction& inst, Thread& thread, FlatMemory& memory) {
    switch (inst.opcode) {
        case Opcode::ADD: {
            int val1 = thread.registers().read(inst.src1);
            int val2 = thread.registers().read(inst.src2);
            thread.registers().write(inst.dst, val1 + val2);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::SUB: {
            int val1 = thread.registers().read(inst.src1);
            int val2 = thread.registers().read(inst.src2);
            thread.registers().write(inst.dst, val1 - val2);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::MUL: {
            int val1 = thread.registers().read(inst.src1);
            int val2 = thread.registers().read(inst.src2);
            thread.registers().write(inst.dst, val1 * val2);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::MOV: {
            // Assume src1 is a register if used this way, or we can use immediate.
            // As per instructions, Day 2 uses basic operands. 
            // We'll treat src1 as register index, but if we want to move immediate, 
            // we can use immediate. Let's look at the instruction format:
            // MOV dst, src1 (we'll implement register-to-register or immediate).
            // Actually, we need to load an immediate, e.g., MOV R0, 5.
            // We'll assume if src1 is -1, it's an immediate move.
            // Let's just use immediate for MOV for simplicity in our straight-line execution test.
            // "MOV dst, immediate" if src1 == -1, else "MOV dst, src1"
            if (inst.src1 == -1) {
                thread.registers().write(inst.dst, inst.immediate);
            } else {
                int val = thread.registers().read(inst.src1);
                thread.registers().write(inst.dst, val);
            }
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::LOAD: {
            // LOAD dst, [immediate]
            int val = memory.load(inst.immediate);
            thread.registers().write(inst.dst, val);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::STORE: {
            // STORE [immediate], src1
            int val = thread.registers().read(inst.src1);
            memory.store(inst.immediate, val);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::CMP: {
            // CMP src1, src2
            int val1 = thread.registers().read(inst.src1);
            int val2 = thread.registers().read(inst.src2);
            thread.set_predicate(val1 == val2);
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::Completed;
        }
        case Opcode::BRANCH: {
            // BRANCH immediate
            if (thread.predicate()) {
                thread.set_pc(thread.pc() + inst.immediate);
                return ExecutionStatus::BranchTaken;
            } else {
                thread.set_pc(thread.pc() + 1);
                return ExecutionStatus::Completed;
            }
        }
        case Opcode::BARRIER: {
            thread.set_pc(thread.pc() + 1);
            return ExecutionStatus::BarrierReached;
        }
        default:
            throw std::runtime_error("Unknown opcode");
    }
}

} // namespace sim_sm
