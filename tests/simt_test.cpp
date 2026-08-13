#include <gtest/gtest.h>
#include "architecture/gpu.hpp"
#include "scheduling/greedy_scheduler.hpp"

using namespace sim_sm;

TEST(SIMTTest, UniformBranch) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) warp.add_thread(sim_sm::Thread(i, 0, i, 0, i));
    sm.add_warp(warp);
    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    // MOV R1, 1; MOV R2, 1; CMP R1, R2; BRANCH taken (uniform)
    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::MOV, 1, -1, -1, 1},
        {sim_sm::Opcode::MOV, 2, -1, -1, 1},
        {sim_sm::Opcode::CMP, -1, 1, 2, 0}, // pred = (R1 == R2) which is always true
        {sim_sm::Opcode::BRANCH, -1, -1, -1, 2},
        {sim_sm::Opcode::MOV, 3, -1, -1, 99}, // Skipped
        {sim_sm::Opcode::MOV, 4, -1, -1, 42}  // Target
    };
    sim_sm::Kernel kernel("uniform", insts);

    sm.tick(kernel, mem); // MOV
    sm.tick(kernel, mem); // MOV
    sm.tick(kernel, mem); // CMP
    sm.tick(kernel, mem); // BRANCH
    
    EXPECT_EQ(sm.get_warps()[0].get_warp_pc(), 5);
    EXPECT_TRUE(sm.get_warps()[0].is_simt_stack_empty());
}

TEST(SIMTTest, SimpleDivergence) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i); // R0 = thread ID
        warp.add_thread(t);
    }
    sm.add_warp(warp);
    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        /* 0 */ {sim_sm::Opcode::SSY, -1, -1, -1, 8}, // Reconvege at 8
        /* 1 */ {sim_sm::Opcode::MOV, 2, -1, -1, 0}, 
        /* 2 */ {sim_sm::Opcode::CMP, -1, 0, 2, 0}, // pred = (R0 == 0)
        /* 3 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 4}, // -> 7 (Thread 0)
        
        /* 4 */ {sim_sm::Opcode::MOV, 1, -1, -1, 1}, // Threads 1-31
        /* 5 */ {sim_sm::Opcode::CMP, -1, 0, 0, 0}, // pred = true
        /* 6 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 2}, // -> 8
        
        /* 7 */ {sim_sm::Opcode::MOV, 1, -1, -1, 2}, // Thread 0
        /* 8 */ {sim_sm::Opcode::SYNC, 0, 0, 0, 0}, // Reconverge
    };
    sim_sm::Kernel kernel("simple", insts);

    for (int i = 0; i < 15; ++i) sm.tick(kernel, mem); // Run to completion

    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
    EXPECT_EQ(sm.get_warps()[0].get_threads()[0].registers().read(1), 2); // Thread 0 executed taken path
    EXPECT_EQ(sm.get_warps()[0].get_threads()[1].registers().read(1), 1); // Thread 1 executed not taken path
}

TEST(SIMTTest, NestedDivergence) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i); 
        warp.add_thread(t);
    }
    sm.add_warp(warp);
    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        /*  0 */ {sim_sm::Opcode::SSY, -1, -1, -1, 16}, // Reconverge at 16 (outer)
        /*  1 */ {sim_sm::Opcode::MOV, 2, -1, -1, 0}, 
        /*  2 */ {sim_sm::Opcode::CMP, -1, 0, 2, 0}, 
        /*  3 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 12}, // -> 15 (Thread 0)
        
        /*  4 */ {sim_sm::Opcode::SSY, -1, -1, -1, 12}, // Reconverge at 12 (inner)
        /*  5 */ {sim_sm::Opcode::MOV, 3, -1, -1, 1}, 
        /*  6 */ {sim_sm::Opcode::CMP, -1, 0, 3, 0}, 
        /*  7 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 4}, // -> 11 (Thread 1)
        
        /*  8 */ {sim_sm::Opcode::MOV, 1, -1, -1, 1}, // Threads 2-31 (Not Taken inner)
        /*  9 */ {sim_sm::Opcode::CMP, -1, 0, 0, 0}, // True
        /* 10 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 2}, // -> 12
        
        /* 11 */ {sim_sm::Opcode::MOV, 1, -1, -1, 2}, // Thread 1 (Taken inner)
        /* 12 */ {sim_sm::Opcode::SYNC, 0, 0, 0, 0}, // Reconverge inner
        
        /* 13 */ {sim_sm::Opcode::CMP, -1, 0, 0, 0}, // True
        /* 14 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 2}, // -> 16 (Threads 1-31 skip outer taken)
        
        /* 15 */ {sim_sm::Opcode::MOV, 1, -1, -1, 3}, // Thread 0 (Taken outer)
        /* 16 */ {sim_sm::Opcode::SYNC, 0, 0, 0, 0}, // Reconverge outer
    };
    
    sim_sm::Kernel kernel("nested", insts);

    for (int i = 0; i < 40; ++i) sm.tick(kernel, mem); // Run to completion

    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
    EXPECT_EQ(sm.get_warps()[0].get_threads()[0].registers().read(1), 3); // Thread 0
    EXPECT_EQ(sm.get_warps()[0].get_threads()[1].registers().read(1), 2); // Thread 1
    EXPECT_EQ(sm.get_warps()[0].get_threads()[2].registers().read(1), 1); // Thread 2
}

TEST(SIMTTest, PartialWarp) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 16; ++i) { // Only 16 threads
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i); 
        warp.add_thread(t);
    }
    sm.add_warp(warp);
    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        /* 0 */ {sim_sm::Opcode::SSY, -1, -1, -1, 8},
        /* 1 */ {sim_sm::Opcode::MOV, 2, -1, -1, 0},
        /* 2 */ {sim_sm::Opcode::CMP, -1, 0, 2, 0}, // pred = (R0 == 0)
        /* 3 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 4}, // -> 7
        /* 4 */ {sim_sm::Opcode::MOV, 1, -1, -1, 1}, 
        /* 5 */ {sim_sm::Opcode::CMP, -1, 0, 0, 0}, 
        /* 6 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 2}, // -> 8
        /* 7 */ {sim_sm::Opcode::MOV, 1, -1, -1, 2}, 
        /* 8 */ {sim_sm::Opcode::SYNC, 0, 0, 0, 0}, 
    };
    sim_sm::Kernel kernel("partial", insts);
    for (int i = 0; i < 15; ++i) sm.tick(kernel, mem); // Run to completion
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
    EXPECT_EQ(sm.get_warps()[0].get_threads()[0].registers().read(1), 2); // Thread 0 executed taken path
    EXPECT_EQ(sm.get_warps()[0].get_threads()[1].registers().read(1), 1); // Thread 1 executed not taken path
}

TEST(SIMTTest, DivergenceWithBarrier) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i);
        warp.add_thread(t);
    }
    sm.add_warp(warp);
    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    // Diverge, Reconverge, then hit BARRIER safely
    std::vector<sim_sm::Instruction> insts = {
        /* 0 */ {sim_sm::Opcode::SSY, -1, -1, -1, 8},
        /* 1 */ {sim_sm::Opcode::MOV, 2, -1, -1, 0},
        /* 2 */ {sim_sm::Opcode::CMP, -1, 0, 2, 0},
        /* 3 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 4}, // -> 7
        /* 4 */ {sim_sm::Opcode::MOV, 1, -1, -1, 1}, 
        /* 5 */ {sim_sm::Opcode::CMP, -1, 0, 0, 0}, 
        /* 6 */ {sim_sm::Opcode::BRANCH, -1, -1, -1, 2}, // -> 8
        /* 7 */ {sim_sm::Opcode::MOV, 1, -1, -1, 2}, 
        /* 8 */ {sim_sm::Opcode::SYNC, 0, 0, 0, 0}, 
        /* 9 */ {sim_sm::Opcode::BARRIER, 0, 0, 0, 0}, // Safe because SYNC restored active_mask
    };
    sim_sm::Kernel kernel("diverge_barrier", insts);
    
    // Ensure the barrier doesn't throw and warp safely completes.
    EXPECT_NO_THROW({
        for (int i = 0; i < 20; ++i) {
            sm.tick(kernel, mem);
        }
    });
    
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
}
