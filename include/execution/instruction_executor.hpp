#pragma once

#include "instruction/instruction.hpp"
#include "architecture/warp.hpp"
#include "memory/memory_system.hpp"

namespace sim_sm {

enum class ExecutionStatus {
    Completed,
    BranchTaken,
    BarrierReached
};

struct ExecutionResult {
    ExecutionStatus status;
    size_t latency;
};

class InstructionExecutor {
public:
    static ExecutionResult execute(const Instruction& inst, Warp& warp, MemorySystem& memory);
};

} // namespace sim_sm
