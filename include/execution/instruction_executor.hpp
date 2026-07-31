#pragma once

#include "instruction/instruction.hpp"
#include "architecture/thread.hpp"
#include "architecture/flat_memory.hpp"

namespace sim_sm {

enum class ExecutionStatus {
    Completed,
    BranchTaken,
    BarrierReached
};

class InstructionExecutor {
public:
    static ExecutionStatus execute(const Instruction& inst, Thread& thread, FlatMemory& memory);
};

} // namespace sim_sm
