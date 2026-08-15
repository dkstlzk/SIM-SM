#pragma once

#include "instruction/instruction.hpp"
#include "architecture/warp.hpp"
#include "memory/memory_system.hpp"
#include <string>

namespace sim_sm {

enum class ExecutionStatus {
    Completed,
    BranchTaken,
    BarrierReached
};

struct ExecutionResult {
    ExecutionStatus status;
    size_t latency;
    size_t memory_transactions = 0;
    std::string memory_space = "";
    size_t write_conflict_stalls = 0;
    size_t bank_conflicts = 0;
    size_t dirty_evictions = 0;
};

class InstructionExecutor {
public:
    static ExecutionResult execute(const Instruction& inst, Warp& warp, MemorySystem& memory);
};

} // namespace sim_sm
