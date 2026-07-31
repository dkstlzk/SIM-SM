#include <gtest/gtest.h>
#include "execution/instruction_executor.hpp"
#include "architecture/thread.hpp"
#include "architecture/flat_memory.hpp"
#include "instruction/instruction.hpp"
#include <stdexcept>

using namespace sim_sm;

class ExecutionTest : public ::testing::Test {
protected:
    Thread thread_{0, 0, 0, 0, 0};
    FlatMemory memory_{100};
};

TEST_F(ExecutionTest, StraightLineExecution) {
    // 5-instruction kernel proving straight-line execution
    // MOV R0, 5
    // MOV R1, 7
    // ADD R2, R0, R1
    // MUL R3, R2, R1
    // SUB R4, R3, R0
    
    std::vector<Instruction> insts = {
        {Opcode::MOV, 0, -1, 0, 5},
        {Opcode::MOV, 1, -1, 0, 7},
        {Opcode::ADD, 2, 0, 1, 0},
        {Opcode::MUL, 3, 2, 1, 0},
        {Opcode::SUB, 4, 3, 0, 0}
    };

    for (const auto& inst : insts) {
        ExecutionStatus status = InstructionExecutor::execute(inst, thread_, memory_);
        EXPECT_EQ(status, ExecutionStatus::Completed);
    }

    EXPECT_EQ(thread_.registers().read(2), 12);
    EXPECT_EQ(thread_.registers().read(3), 84);
    EXPECT_EQ(thread_.registers().read(4), 79);
}

TEST_F(ExecutionTest, LoadStoreExecution) {
    // STORE [10], R0 (assume R0 has 42)
    // LOAD R1, [10]
    thread_.registers().write(0, 42);

    Instruction store_inst = {Opcode::STORE, 0, 0, 0, 10};
    Instruction load_inst = {Opcode::LOAD, 1, 0, 0, 10};

    EXPECT_EQ(InstructionExecutor::execute(store_inst, thread_, memory_), ExecutionStatus::Completed);
    EXPECT_EQ(InstructionExecutor::execute(load_inst, thread_, memory_), ExecutionStatus::Completed);

    EXPECT_EQ(thread_.registers().read(1), 42);
}

TEST_F(ExecutionTest, MemoryBoundsCheck) {
    Instruction load_inst = {Opcode::LOAD, 0, 0, 0, 999};
    EXPECT_THROW(InstructionExecutor::execute(load_inst, thread_, memory_), std::out_of_range);

    Instruction store_inst = {Opcode::STORE, 0, 0, 0, 999};
    EXPECT_THROW(InstructionExecutor::execute(store_inst, thread_, memory_), std::out_of_range);
}

TEST_F(ExecutionTest, BranchExecutionTaken) {
    thread_.registers().write(0, 10);
    thread_.registers().write(1, 10);
    thread_.set_pc(5);

    Instruction cmp_inst = {Opcode::CMP, 0, 0, 1, 0};
    EXPECT_EQ(InstructionExecutor::execute(cmp_inst, thread_, memory_), ExecutionStatus::Completed);
    EXPECT_TRUE(thread_.predicate());
    EXPECT_EQ(thread_.pc(), 6);

    Instruction branch_inst = {Opcode::BRANCH, 0, 0, 0, 2};
    EXPECT_EQ(InstructionExecutor::execute(branch_inst, thread_, memory_), ExecutionStatus::BranchTaken);
    // PC was 6 before BRANCH execution, branching +2 means new PC = 8
    EXPECT_EQ(thread_.pc(), 8);
}

TEST_F(ExecutionTest, BranchExecutionNotTaken) {
    thread_.registers().write(0, 10);
    thread_.registers().write(1, 20); // Not equal
    thread_.set_pc(5);

    Instruction cmp_inst = {Opcode::CMP, 0, 0, 1, 0};
    EXPECT_EQ(InstructionExecutor::execute(cmp_inst, thread_, memory_), ExecutionStatus::Completed);
    EXPECT_FALSE(thread_.predicate());
    EXPECT_EQ(thread_.pc(), 6);

    Instruction branch_inst = {Opcode::BRANCH, 0, 0, 0, 2};
    EXPECT_EQ(InstructionExecutor::execute(branch_inst, thread_, memory_), ExecutionStatus::Completed);
    // PC was 6, branch not taken -> PC = 7
    EXPECT_EQ(thread_.pc(), 7);
}

TEST_F(ExecutionTest, BarrierExecution) {
    Instruction barrier_inst = {Opcode::BARRIER, 0, 0, 0, 0};
    EXPECT_EQ(InstructionExecutor::execute(barrier_inst, thread_, memory_), ExecutionStatus::BarrierReached);
}
