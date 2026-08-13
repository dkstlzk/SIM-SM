#include <gtest/gtest.h>
#include "execution/instruction_executor.hpp"
#include "architecture/thread.hpp"
#include "architecture/flat_memory.hpp"
#include "instruction/instruction.hpp"
#include <stdexcept>

#include "memory/memory_system.hpp"
#include "memory/cache.hpp"
#include "memory/shared_memory.hpp"
#include "memory/global_memory.hpp"
#include "architecture/warp.hpp"

using namespace sim_sm;

class ExecutionTest : public ::testing::Test {
protected:
    ExecutionTest() : memory_(shared_memory_, l1_cache_, l2_cache_, global_memory_) {
        warp_.add_thread(Thread(0, 0, 0, 0, 0));
    }

    Warp warp_{0};
    SharedMemory shared_memory_{65536};
    Cache l1_cache_{4, 4, 32};
    Cache l2_cache_{16, 8, 32};
    GlobalMemory global_memory_{1048576};
    MemorySystem memory_;

    Thread& get_thread() { return warp_.get_threads()[0]; }
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
        ExecutionResult result = InstructionExecutor::execute(inst, warp_, memory_);
        EXPECT_EQ(result.status, ExecutionStatus::Completed);
        warp_.set_warp_pc(warp_.get_warp_pc() + 1);
    }

    EXPECT_EQ(get_thread().registers().read(2), 12);
    EXPECT_EQ(get_thread().registers().read(3), 84);
    EXPECT_EQ(get_thread().registers().read(4), 79);
}

TEST_F(ExecutionTest, LoadStoreExecution) {
    // STORE [10], R0 (assume R0 has 42)
    // LOAD R1, [10]
    get_thread().registers().write(0, 42);

    Instruction store_inst = {Opcode::STORE, 0, 0, 0, 10};
    Instruction load_inst = {Opcode::LOAD, 1, 0, 0, 10};

    EXPECT_EQ(InstructionExecutor::execute(store_inst, warp_, memory_).status, ExecutionStatus::Completed);
    warp_.set_warp_pc(warp_.get_warp_pc() + 1);
    EXPECT_EQ(InstructionExecutor::execute(load_inst, warp_, memory_).status, ExecutionStatus::Completed);

    EXPECT_EQ(get_thread().registers().read(1), 42);
}

TEST_F(ExecutionTest, MemoryBoundsCheck) {
    // Note: With GlobalMemory of size 1MB, 1048576 is out of bounds
    Instruction load_inst = {Opcode::LOAD, 0, 0, 0, static_cast<int>(1048576)};
    EXPECT_THROW(InstructionExecutor::execute(load_inst, warp_, memory_), std::out_of_range);

    Instruction store_inst = {Opcode::STORE, 0, 0, 0, static_cast<int>(1048576)};
    EXPECT_THROW(InstructionExecutor::execute(store_inst, warp_, memory_), std::out_of_range);
}

TEST_F(ExecutionTest, BranchExecutionTaken) {
    get_thread().registers().write(0, 10);
    get_thread().registers().write(1, 10);
    get_thread().set_pc(5);
    warp_.set_warp_pc(5);

    Instruction cmp_inst = {Opcode::CMP, 0, 0, 1, 0};
    EXPECT_EQ(InstructionExecutor::execute(cmp_inst, warp_, memory_).status, ExecutionStatus::Completed);
    EXPECT_TRUE(get_thread().predicate());
    EXPECT_EQ(get_thread().pc(), 6);

    Instruction branch_inst = {Opcode::BRANCH, 0, 0, 0, 2};
    EXPECT_EQ(InstructionExecutor::execute(branch_inst, warp_, memory_).status, ExecutionStatus::BranchTaken);
    // PC was 6 before BRANCH execution, branching +2 means new PC = 8
    EXPECT_EQ(get_thread().pc(), 8);
}

TEST_F(ExecutionTest, BranchExecutionNotTaken) {
    get_thread().registers().write(0, 10);
    get_thread().registers().write(1, 20); // Not equal
    get_thread().set_pc(5);
    warp_.set_warp_pc(5);

    Instruction cmp_inst = {Opcode::CMP, 0, 0, 1, 0};
    EXPECT_EQ(InstructionExecutor::execute(cmp_inst, warp_, memory_).status, ExecutionStatus::Completed);
    EXPECT_FALSE(get_thread().predicate());
    EXPECT_EQ(get_thread().pc(), 6);

    Instruction branch_inst = {Opcode::BRANCH, 0, 0, 0, 2};
    EXPECT_EQ(InstructionExecutor::execute(branch_inst, warp_, memory_).status, ExecutionStatus::Completed);
    // PC was 6, branch not taken -> PC = 7
    EXPECT_EQ(get_thread().pc(), 7);
}

TEST_F(ExecutionTest, BarrierExecution) {
    Instruction barrier_inst = {Opcode::BARRIER, 0, 0, 0, 0};
    EXPECT_EQ(InstructionExecutor::execute(barrier_inst, warp_, memory_).status, ExecutionStatus::BarrierReached);
}

TEST_F(ExecutionTest, CoalescedLoadExecution) {
    Warp full_warp(0);
    for (int i = 0; i < 32; ++i) {
        Thread t(i, 0, 0, 0, 0);
        // Base address in R0 is 0
        t.registers().write(0, 0);
        full_warp.add_thread(t);
    }

    // LOAD R1, [R0 + (thread_id * 4)] is what we want, but wait...
    // Our ISA doesn't have thread_id.
    // We can pre-load R0 with different values for each thread!
    for (int i = 0; i < 32; ++i) {
        full_warp.get_threads()[i].registers().write(0, i * 4);
    }

    Instruction load_inst = {Opcode::LOAD, 1, 0, -1, 0};
    ExecutionResult result = InstructionExecutor::execute(load_inst, full_warp, memory_);

    // Line size is 32 bytes (set in test setup).
    // 32 threads * 4 bytes = 128 bytes.
    // 128 bytes / 32 bytes = 4 cache lines = 4 transactions.
    EXPECT_EQ(result.memory_transactions, 4);

    // Latency = base latency (e.g. L1 hit is 1 cycle, or miss is 10 cycles etc.)
    // If it's a miss, base latency = L1_miss_penalty + L1_hit_latency = 10 + 1 = 11?
    // Let's assume memory access gives some max_latency, and total = max_latency + (4 - 1).
    // The exact cycle count depends on the cache miss/hit state.
    // We just verify the transaction count for now, and that latency is reasonable.
    EXPECT_GT(result.latency, 3);
}

TEST_F(ExecutionTest, StridedLoadExecution) {
    Warp full_warp(0);
    for (int i = 0; i < 32; ++i) {
        Thread t(i, 0, 0, 0, 0);
        t.registers().write(0, i * 32); // stride of 32 bytes
        full_warp.add_thread(t);
    }

    Instruction load_inst = {Opcode::LOAD, 1, 0, -1, 0};
    ExecutionResult result = InstructionExecutor::execute(load_inst, full_warp, memory_);

    // 32 threads, each in a different 32-byte cache line = 32 transactions.
    EXPECT_EQ(result.memory_transactions, 32);
}
