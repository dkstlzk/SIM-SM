#include <gtest/gtest.h>
#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/thread_block.hpp"
#include "architecture/warp.hpp"
#include "architecture/thread.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include "scheduling/greedy_scheduler.hpp"

namespace {

sim_sm::Grid build_grid(size_t num_threads, size_t block_size, size_t warp_size, size_t& total_warps) {
    sim_sm::Grid grid;
    total_warps = 0;

    size_t num_blocks = (num_threads + block_size - 1) / block_size;
    if (num_threads == 0) {
        num_blocks = 0;
    }

    for (size_t b = 0; b < num_blocks; ++b) {
        sim_sm::ThreadBlock block(b);

        size_t threads_in_this_block = std::min(block_size, num_threads - b * block_size);
        size_t warps_in_this_block = (threads_in_this_block + warp_size - 1) / warp_size;

        for (size_t w = 0; w < warps_in_this_block; ++w) {
            size_t local_warp_id = w;
            sim_sm::Warp warp(local_warp_id);

            size_t threads_in_this_warp = std::min(warp_size, threads_in_this_block - w * warp_size);

            for (size_t l = 0; l < threads_in_this_warp; ++l) {
                size_t local_id = w * warp_size + l;
                size_t global_id = b * block_size + local_id;
                size_t lane_id = l;
                sim_sm::Thread thread(global_id, b, local_id, local_warp_id, lane_id);
                warp.add_thread(thread);
            }

            block.add_warp(warp);
            total_warps++;
        }
        grid.add_block(block);
    }
    return grid;
}

} // namespace

TEST(ArchitectureTest, ExactMultipleOfWarpAndBlock) {
    size_t total_warps = 0;
    sim_sm::Grid grid = build_grid(1024, 128, 32, total_warps);

    EXPECT_EQ(grid.get_blocks().size(), 8);
    EXPECT_EQ(total_warps, 32);

    auto& blocks = grid.get_blocks();
    EXPECT_EQ(blocks[0].get_warps().size(), 4);
    EXPECT_EQ(blocks[0].get_warps()[0].get_threads().size(), 32);

    auto& thread = blocks[1].get_warps()[1].get_threads()[2];
    EXPECT_EQ(thread.get_block_id(), 1);
    EXPECT_EQ(thread.get_warp_id(), 1);
    EXPECT_EQ(thread.get_lane_id(), 2);
    EXPECT_EQ(thread.get_local_id(), 34); // 1*32 + 2 = 34
    EXPECT_EQ(thread.get_global_id(), 162); // 1*128 + 34 = 162
}

TEST(ArchitectureTest, PartialWarp) {
    size_t total_warps = 0;
    sim_sm::Grid grid = build_grid(10, 128, 32, total_warps);

    EXPECT_EQ(grid.get_blocks().size(), 1);
    EXPECT_EQ(total_warps, 1);

    auto& blocks = grid.get_blocks();
    EXPECT_EQ(blocks[0].get_warps().size(), 1);
    EXPECT_EQ(blocks[0].get_warps()[0].get_threads().size(), 10);

    auto& thread = blocks[0].get_warps()[0].get_threads()[9];
    EXPECT_EQ(thread.get_global_id(), 9);
    EXPECT_EQ(thread.get_lane_id(), 9);
}

TEST(ArchitectureTest, PartialBlock) {
    size_t total_warps = 0;
    sim_sm::Grid grid = build_grid(150, 128, 32, total_warps);

    EXPECT_EQ(grid.get_blocks().size(), 2);

    auto& blocks = grid.get_blocks();
    // Block 0: 128 threads -> 4 warps
    EXPECT_EQ(blocks[0].get_warps().size(), 4);
    // Block 1: 22 threads -> 1 warp
    EXPECT_EQ(blocks[1].get_warps().size(), 1);
    EXPECT_EQ(blocks[1].get_warps()[0].get_threads().size(), 22);
    EXPECT_EQ(total_warps, 5);
}

TEST(ArchitectureTest, EmptyGrid) {
    size_t total_warps = 0;
    sim_sm::Grid grid = build_grid(0, 128, 32, total_warps);

    EXPECT_EQ(grid.get_blocks().size(), 0);
    EXPECT_EQ(total_warps, 0);
}

#include "architecture/occupancy.hpp"

using namespace sim_sm;

TEST(OccupancyTest, MaxThreadsLimit) {
    SystemConfig config = {4, 32, 256, 2048, 32, 65536, 65536};
    KernelResourceRequirements req = {10, 1024};
    // 256 threads -> max blocks by threads = 2048 / 256 = 8

    OccupancyResult result = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(result.resident_blocks, 8);
    EXPECT_EQ(result.resident_warps, 8 * (256/32)); // 8 * 8 = 64
    EXPECT_DOUBLE_EQ(result.occupancy_percentage, 64.0 / (2048/32)); // 64 / 64 = 1.0
}

TEST(OccupancyTest, MaxSharedMemoryLimit) {
    SystemConfig config = {4, 32, 128, 2048, 32, 65536, 65536};
    KernelResourceRequirements req = {10, 32768}; // 32KB per block
    // blocks by memory = 65536 / 32768 = 2

    OccupancyResult result = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(result.resident_blocks, 2);
    EXPECT_EQ(result.resident_warps, 2 * (128/32)); // 8
    EXPECT_DOUBLE_EQ(result.occupancy_percentage, 8.0 / 64.0);
}

TEST(OccupancyTest, MaxRegistersLimit) {
    SystemConfig config = {4, 32, 128, 2048, 32, 65536, 65536};
    KernelResourceRequirements req = {256, 0}; // 128 threads, 256 regs/thread -> 32768 regs/block
    // blocks by registers = 65536 / 32768 = 2

    OccupancyResult result = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(result.resident_blocks, 2);
}

TEST(GPUTest, MultipleWavesDispatch) {
    SystemConfig config = {1, 32, 128, 2048, 32, 65536, 65536};
    GPU gpu(1, 4, 4, 16, 8, 32, 1048576);

    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<RoundRobinScheduler>());
    }

    // 10 blocks of 128 threads.
    size_t total_warps = 0;
    Grid grid = build_grid(1280, 128, 32, total_warps);
    EXPECT_EQ(grid.get_blocks().size(), 10);

    // Limit resident blocks to 2
    KernelResourceRequirements req = {10, 32768}; // 32KB shared mem per block -> 65536/32768 = 2 blocks per SM

    // Simple kernel with 1 instruction
    Kernel kernel("dummy", {{Opcode::BARRIER, 0, 0, 0, 0}});

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    size_t total_insts = 0;
    for (const auto& sm : gpu.get_sms()) {
        total_insts += sm.get_counters().get_instructions_retired();
    }
    EXPECT_EQ(total_insts, total_warps);
}

TEST(SMTest, ForwardDivergence) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i % 2 == 0 ? 1 : 0); // R0 is predicate
        warp.add_thread(t);
    }
    sm.add_warp(warp);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm); // dummy

    // R0 is 1 for even, 0 for odd
    // MOV R1, 1
    // CMP R0 == R1 -> predicate
    // BRANCH +2 if true
    // MOV R2, 10
    // MOV R3, 0
    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::MOV, 1, -1, 0, 1},     // 0: MOV R1, 1
        {sim_sm::Opcode::CMP, 0, 0, 1, 0},      // 1: CMP R0 == R1 (is even?)
        {sim_sm::Opcode::BRANCH, 0, 0, 0, 2},   // 2: if even, jump to 4
        {sim_sm::Opcode::MOV, 2, -1, 0, 10},    // 3: odd path, R2 = 10
        {sim_sm::Opcode::MOV, 3, -1, 0, 0},     // 4: both reconverge here! MOV R3, 0
    };
    sim_sm::Kernel kernel("test", insts);

    while(!sm.is_completed()) {
        sm.tick(kernel, mem);
    }

    auto& final_warp = sm.get_warps()[0];
    for (int i = 0; i < 32; ++i) {
        if (i % 2 == 0) {
            // Even path skipped MOV R2, 10
            EXPECT_EQ(final_warp.get_threads()[i].registers().read(2), 0);
        } else {
            EXPECT_EQ(final_warp.get_threads()[i].registers().read(2), 10);
        }
        // Both hit 4
        EXPECT_EQ(final_warp.get_threads()[i].pc(), 5);
    }
}

TEST(SMTest, BarrierSingleWarp) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        warp.add_thread(t);
    }
    sm.add_warp(warp);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0},
        {sim_sm::Opcode::ADD, 0, 0, 0, 0} // Dummy instruction after barrier
    };
    sim_sm::Kernel kernel("test", insts);

    // Tick once, it should hit barrier, block has 1 warp, so it immediately releases back to Ready
    sm.tick(kernel, mem);
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Ready);

    // Tick again, it executes the ADD and Completes
    sm.tick(kernel, mem);
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
}

TEST(SMTest, BarrierMultiWarpSameBlock) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());

    sim_sm::Warp w0(0);
    for (int i = 0; i < 32; ++i) w0.add_thread(sim_sm::Thread(i, 0, i, 0, i));
    sim_sm::Warp w1(1);
    for (int i = 0; i < 32; ++i) w1.add_thread(sim_sm::Thread(32+i, 0, 32+i, 1, i)); // Same block 0
    sm.add_warp(w0);
    sm.add_warp(w1);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0}
    };
    sim_sm::Kernel kernel("test", insts);

    sm.tick(kernel, mem); // Warp 0 hits barrier -> StalledAtBarrier
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::StalledAtBarrier);
    EXPECT_EQ(sm.get_warps()[1].get_state(), sim_sm::WarpState::Ready);

    sm.tick(kernel, mem); // Warp 1 hits barrier -> Both released -> Warp 1 Completes, Warp 0 becomes Ready
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Ready);
    EXPECT_EQ(sm.get_warps()[1].get_state(), sim_sm::WarpState::Completed);
}

TEST(SMTest, BarrierDifferentBlocksIndependent) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());

    sim_sm::Warp w0(0);
    for (int i = 0; i < 32; ++i) w0.add_thread(sim_sm::Thread(i, 0, i, 0, i)); // Block 0
    sim_sm::Warp w1(1);
    for (int i = 0; i < 32; ++i) w1.add_thread(sim_sm::Thread(32+i, 1, i, 1, i)); // Block 1
    sm.add_warp(w0);
    sm.add_warp(w1);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0}
    };
    sim_sm::Kernel kernel("test", insts);

    sm.tick(kernel, mem); // Warp 0 hits barrier. It's the only warp in block 0, so it releases and completes immediately!
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Completed);
}

TEST(SMTest, DivergentBarrierWithinWarp) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(0, i % 2 == 0 ? 1 : 0);
        warp.add_thread(t);
    }
    sm.add_warp(warp);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::MOV, 1, -1, 0, 1},
        {sim_sm::Opcode::CMP, 0, 0, 1, 0},
        {sim_sm::Opcode::BRANCH, 0, 0, 0, 2},
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0}, // Odd path hits barrier
        {sim_sm::Opcode::MOV, 3, -1, 0, 0},
    };
    sim_sm::Kernel kernel("test", insts);

    // Tick MOV
    sm.tick(kernel, mem);
    // Tick CMP
    sm.tick(kernel, mem);
    // Tick BRANCH
    sm.tick(kernel, mem);
    // Tick BARRIER (should throw)
    EXPECT_THROW(sm.tick(kernel, mem), std::runtime_error);
}

TEST(SMTest, MalformedBarrierCompletedWarp) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());

    // We want Warp 1 to complete FIRST, and then Warp 0 hits a barrier.
    sim_sm::Warp w0(0); // This warp will hit barrier
    for (int i = 0; i < 32; ++i) w0.add_thread(sim_sm::Thread(i, 0, i, 0, i));

    sim_sm::Warp w1(1); // This warp will just complete
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(32+i, 0, 32+i, 1, i);
        t.set_pc(1); // Skip the barrier instruction!
        w1.add_thread(t);
    }
    w1.set_warp_pc(1);

    // Add w1 first so it executes first (Greedy scheduler picks first ready)
    sm.add_warp(w1);
    sm.add_warp(w0);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0},
        {sim_sm::Opcode::MOV, 3, -1, 0, 0}
    };
    sim_sm::Kernel kernel("test", insts);

    // Tick Warp 1 -> executes MOV
    sm.tick(kernel, mem);
    // At this point, Warp 1 sets its PC to 2 and is Completed!

    // Now Warp 0 executes -> hits BARRIER.
    // It sees Warp 1 is in same block but Completed. It should throw Malformed barrier.
    EXPECT_THROW(sm.tick(kernel, mem), std::runtime_error);
}

TEST(SMTest, BarrierBlockWithThreeWarps) {
    sim_sm::SM sm(0);
    sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());

    sim_sm::Warp w0(0);
    for (int i = 0; i < 32; ++i) w0.add_thread(sim_sm::Thread(i, 0, i, 0, i)); // Block 0
    sim_sm::Warp w1(1);
    for (int i = 0; i < 32; ++i) w1.add_thread(sim_sm::Thread(32+i, 0, 32+i, 1, i)); // Block 0
    sim_sm::Warp w2(2);
    for (int i = 0; i < 32; ++i) w2.add_thread(sim_sm::Thread(64+i, 0, 64+i, 2, i)); // Block 0

    sm.add_warp(w0);
    sm.add_warp(w1);
    sm.add_warp(w2);

    sim_sm::GlobalMemory gm(1024);
    sim_sm::MemorySystem mem(sm.get_shared_memory(), sm.get_l1_cache(), sm.get_l1_cache(), gm);

    // Instruction: BARRIER
    std::vector<sim_sm::Instruction> insts = {
        {sim_sm::Opcode::BARRIER, 0, 0, 0, 0}
    };
    sim_sm::Kernel kernel("test", insts);

    sm.tick(kernel, mem); // Warp 0 -> barrier
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::StalledAtBarrier);
    EXPECT_EQ(sm.get_warps()[1].get_state(), sim_sm::WarpState::Ready);
    EXPECT_EQ(sm.get_warps()[2].get_state(), sim_sm::WarpState::Ready);

    sm.tick(kernel, mem); // Warp 1 -> barrier
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::StalledAtBarrier);
    EXPECT_EQ(sm.get_warps()[1].get_state(), sim_sm::WarpState::StalledAtBarrier);
    EXPECT_EQ(sm.get_warps()[2].get_state(), sim_sm::WarpState::Ready);

    sm.tick(kernel, mem); // Warp 2 -> barrier. All release. Warp 2 completes immediately as it executes the barrier.
    EXPECT_EQ(sm.get_warps()[0].get_state(), sim_sm::WarpState::Ready);
    EXPECT_EQ(sm.get_warps()[1].get_state(), sim_sm::WarpState::Ready);
    EXPECT_EQ(sm.get_warps()[2].get_state(), sim_sm::WarpState::Completed);
}

