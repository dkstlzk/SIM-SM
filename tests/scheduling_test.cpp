#include <gtest/gtest.h>
#include "architecture/sm.hpp"
#include "architecture/warp.hpp"
#include "architecture/kernel.hpp"
#include "scheduling/greedy_scheduler.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include "memory/memory_system.hpp"
#include "memory/cache.hpp"
#include "memory/shared_memory.hpp"
#include "memory/global_memory.hpp"

using namespace sim_sm;

class SchedulingTest : public ::testing::Test {
protected:
    Kernel create_controlled_workload() {
        std::vector<Instruction> insts = {
            {Opcode::MOV, 0, -1, 0, 5},
            {Opcode::LOAD, 1, 0, 0, 10}, // Synthetic latency 5
            {Opcode::ADD, 2, 0, 1, 0},
            {Opcode::MUL, 3, 2, 1, 0},   // Synthetic latency 5
            {Opcode::STORE, 0, 0, 0, 20} // Synthetic latency 5
        };
        return Kernel("controlled_workload", insts);
    }

    void setup_sm(SM& sm, size_t num_warps) {
        for (size_t i = 0; i < num_warps; ++i) {
            Warp warp(i);
            // 32 threads
            for (size_t t = 0; t < 32; ++t) {
                warp.add_thread(Thread(i, t, 0, 0, 0));
            }
            sm.add_warp(warp);
        }
    }

    SharedMemory shared_mem_{65536};
    Cache l1_cache_{4, 4, 32};
    Cache l2_cache_{16, 8, 32};
    GlobalMemory global_mem_{1048576};

    std::unique_ptr<MemorySystem> create_memory_system() {
        MemoryAccessConfig config;
        config.l1_latency = 5;
        config.l2_latency = 0;
        config.global_memory_latency = 0; // Total latency 5 cycles regardless of hit/miss
        return std::make_unique<MemorySystem>(shared_mem_, l1_cache_, l2_cache_, global_mem_, config);
    }
};

TEST_F(SchedulingTest, GreedySchedulerUnit) {
    std::vector<Warp> warps;
    warps.emplace_back(0);
    warps.emplace_back(1);
    warps.emplace_back(2);

    GreedyScheduler scheduler;

    // First ready should be 0
    EXPECT_EQ(scheduler.select_warp(warps)->get_warp_id(), 0);

    // Stall warp 0
    warps[0].stall(5);
    EXPECT_EQ(scheduler.select_warp(warps)->get_warp_id(), 1);

    // Stall warp 1
    warps[1].stall(5);
    EXPECT_EQ(scheduler.select_warp(warps)->get_warp_id(), 2);

    // Stall warp 2
    warps[2].stall(5);
    EXPECT_EQ(scheduler.select_warp(warps), nullptr);
}

TEST_F(SchedulingTest, RoundRobinSchedulerUnit) {
    std::vector<Warp> warps;
    warps.emplace_back(0);
    warps.emplace_back(1);
    warps.emplace_back(2);
    RoundRobinScheduler scheduler;

    // Initially next is 0
    Warp* selected = scheduler.select_warp(warps);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_warp_id(), 0);

    // Next should be 1
    selected = scheduler.select_warp(warps);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_warp_id(), 1);

    // Next should be 2
    selected = scheduler.select_warp(warps);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_warp_id(), 2);

    // Next should wrap around to 0
    selected = scheduler.select_warp(warps);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_warp_id(), 0);

    // Stall warp 1
    warps[1].stall(5);

    // Next should skip 1 and select 2
    EXPECT_EQ(scheduler.select_warp(warps)->get_warp_id(), 2);
}

TEST_F(SchedulingTest, FinalInstructionCompletesWarp) {
    SM sm(0);
    sm.set_scheduler(std::make_unique<GreedyScheduler>());

    Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        warp.add_thread(Thread(i, i, 0, 0, 0));
    }
    sm.add_warp(warp);

    // Kernel with a single LOAD instruction
    std::vector<Instruction> insts = {
        {Opcode::LOAD, 1, 0, 0, 10}
    };
    Kernel kernel("single_load", insts);

    auto memory = create_memory_system();

    while (!sm.is_completed()) {
        sm.tick(kernel, *memory);
    }

    for (const auto& w : sm.get_warps()) {
        EXPECT_EQ(w.get_state(), WarpState::Completed);
        EXPECT_EQ(w.get_warp_pc(), kernel.instructions().size());
    }
}

TEST_F(SchedulingTest, SMExecutionIntegration) {
    Kernel kernel = create_controlled_workload();
    auto memory = create_memory_system();

    SM sm_greedy(0);
    setup_sm(sm_greedy, 2);
    sm_greedy.set_scheduler(std::make_unique<GreedyScheduler>());

    SM sm_rr(1);
    setup_sm(sm_rr, 2);
    sm_rr.set_scheduler(std::make_unique<RoundRobinScheduler>());

    // Run both to completion
    while (!sm_greedy.is_completed()) {
        sm_greedy.tick(kernel, *memory);
    }

    while (!sm_rr.is_completed()) {
        sm_rr.tick(kernel, *memory);
    }

    // Both should have retired 10 instructions (5 per warp * 2 warps)
    EXPECT_EQ(sm_greedy.get_counters().get_instructions_retired(), 10);
    EXPECT_EQ(sm_rr.get_counters().get_instructions_retired(), 10);

    size_t greedy_cycles = sm_greedy.get_counters().get_cycles();
    size_t rr_cycles = sm_rr.get_counters().get_cycles();

    // Just assert that both complete and log their performance counters
    // We observe deterministic cycle counts for this specific workload
    // Cycles changed because MUL no longer has synthetic latency, and stall is now (latency - 1)
    // 2 warps, 5 instructions each.
    // Latencies: MOV(1), LOAD(5 -> stall 4), ADD(1), MUL(1), STORE(5 -> stall 4).
    EXPECT_EQ(greedy_cycles, 11);
    EXPECT_EQ(rr_cycles, 12);

    // Verify IPC calculation
    double rr_ipc = sm_rr.get_counters().get_ipc();
    EXPECT_GT(rr_ipc, 0.0);

    double greedy_ipc = sm_greedy.get_counters().get_ipc();
    EXPECT_GT(greedy_ipc, 0.0);
}
