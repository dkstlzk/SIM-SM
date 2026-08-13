#include <gtest/gtest.h>
#include "architecture/occupancy.hpp"
#include "architecture/sm.hpp"
#include "architecture/gpu.hpp"
#include "architecture/thread_block.hpp"
#include "scheduling/round_robin_scheduler.hpp"

using namespace sim_sm;

// Helper to create a dummy block
ThreadBlock create_dummy_block(size_t block_id, size_t num_threads, size_t warp_size) {
    ThreadBlock block(block_id);
    size_t thread_id = 0;
    while (thread_id < num_threads) {
        size_t warp_id = thread_id / warp_size;
        Warp warp(warp_id);
        size_t threads_in_warp = std::min(warp_size, num_threads - thread_id);
        for (size_t i = 0; i < threads_in_warp; ++i) {
            warp.add_thread(Thread(thread_id + i, block_id, i, warp_id, i));
        }
        block.add_warp(warp);
        thread_id += threads_in_warp;
    }
    return block;
}

TEST(OccupancyTest, RegisterLimited) {
    SystemConfig config{1, 32, 256, 2048, 16, 65536, 65536};
    KernelResourceRequirements req{64, 0}; // 64 regs/thread -> 16384 per block. 65536 / 16384 = 4 blocks.
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 4);
    EXPECT_EQ(occ.limiting_factor, "Registers");
}

TEST(OccupancyTest, SharedMemoryLimited) {
    SystemConfig config{1, 32, 256, 2048, 16, 65536, 65536};
    KernelResourceRequirements req{16, 32768}; // 32768 shmem/block. 65536 / 32768 = 2 blocks.
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 2);
    EXPECT_EQ(occ.limiting_factor, "Shared Memory");
}

TEST(OccupancyTest, WarpVsThreadTie) {
    SystemConfig config{1, 32, 512, 1024, 16, 65536, 65536, 32}; // Max 1024 threads/SM -> 2 blocks of 512. Tie with warps: 32 max warps.
    KernelResourceRequirements req{0, 0};
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 2);
    EXPECT_EQ(occ.limiting_factor, "Warps");
}

TEST(OccupancyTest, ThreadsLimited) {
    // Force a scenario where Warps is explicitly not the limit.
    // E.g., max_warps_per_sm = 64. max_threads = 64. block_size = 64.
    // 64/64 = 1 block. warps=2. 64 warps max. Thread capacity uniquely limits it.
    SystemConfig config{1, 32, 64, 64, 16, 65536, 65536, 64}; // Explicit max_warps_per_sm = 64
    KernelResourceRequirements req{0, 0};
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 1);
    EXPECT_EQ(occ.limiting_factor, "Threads");
}

TEST(OccupancyTest, BlocksLimited) {
    SystemConfig config{1, 32, 32, 2048, 4, 65536, 65536}; // Max 4 blocks/SM
    KernelResourceRequirements req{0, 0};
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 4);
    EXPECT_EQ(occ.limiting_factor, "Blocks");
}

TEST(OccupancyTest, ExactTiePriority) {
    SystemConfig config{1, 32, 256, 1024, 4, 16384, 65536}; // 4 blocks allows 1024 threads, 4 blocks.
    KernelResourceRequirements req{16, 4096};
    // Regs: 256*16=4096. 65536/4096 = 16.
    // Shmem: 4096. 16384/4096 = 4.
    // Threads: 1024/256 = 4.
    // Blocks: 4.
    // Tie between Shmem, Threads, Blocks. Shmem has highest priority.
    auto occ = OccupancyCalculator::compute(config, req);
    EXPECT_EQ(occ.resident_blocks, 4);
    EXPECT_EQ(occ.limiting_factor, "Shared Memory");
}

TEST(OccupancyTest, PartialFinalWarp) {
    SystemConfig config{1, 32, 50, 2048, 16, 65536, 65536}; // 50 threads/block -> 2 warps/block.
    KernelResourceRequirements req{10, 0};
    auto occ = OccupancyCalculator::compute(config, req);
    // 50 * 10 = 500 regs/block.
    // 65536 / 500 = 131 blocks.
    // threads: 2048 / 50 = 40 blocks.
    // warps: 64 max warps / 2 warps_per_block = 32 blocks.
    // Blocks: 16
    // Limiting is Blocks.
    EXPECT_EQ(occ.resident_blocks, 16);
    EXPECT_EQ(occ.resident_warps, 32);
    EXPECT_EQ(occ.limiting_factor, "Blocks");
}

TEST(OccupancyTest, ZeroConfigTest) {
    SystemConfig config{1, 0, 0, 2048, 16, 65536, 65536};
    KernelResourceRequirements req{0, 0};
    EXPECT_THROW(OccupancyCalculator::compute(config, req), std::invalid_argument);
}

TEST(SMResidencyTest, DynamicAllocationAndRelease) {
    SM sm(0);
    SystemConfig config{1, 32, 64, 2048, 4, 65536, 65536}; // Max 4 blocks.
    KernelResourceRequirements req{10, 100}; // 64*10=640 regs, 100 shmem per block

    // Allocate until exhausted
    for (size_t i = 0; i < 4; ++i) {
        ThreadBlock block = create_dummy_block(i, 64, 32);
        EXPECT_TRUE(sm.can_admit(block, config, req));
        sm.allocate_block(block, config, req);
    }

    EXPECT_EQ(sm.get_allocated_blocks(), 4);
    EXPECT_EQ(sm.get_allocated_warps(), 8);
    EXPECT_EQ(sm.get_allocated_threads(), 256);
    EXPECT_EQ(sm.get_allocated_registers(), 2560);
    EXPECT_EQ(sm.get_allocated_shared_memory(), 400);
    EXPECT_EQ(sm.get_resident_blocks().size(), 4);

    // 5th block should be rejected
    ThreadBlock block5 = create_dummy_block(4, 64, 32);
    EXPECT_FALSE(sm.can_admit(block5, config, req));

    // Complete block 0
    for (auto& w : sm.get_warps()) {
        if (!w.get_threads().empty() && w.get_threads()[0].get_block_id() == 0) {
            w.set_completed();
        }
    }

    // Release resources
    sm.release_completed_blocks();

    EXPECT_EQ(sm.get_allocated_blocks(), 3);
    EXPECT_EQ(sm.get_allocated_warps(), 6);
    EXPECT_EQ(sm.get_allocated_threads(), 192);
    EXPECT_EQ(sm.get_allocated_registers(), 1920);
    EXPECT_EQ(sm.get_allocated_shared_memory(), 300);
    EXPECT_EQ(sm.get_resident_blocks().size(), 3);

    // Now block 5 should be admitted
    EXPECT_TRUE(sm.can_admit(block5, config, req));
    sm.allocate_block(block5, config, req);
}

TEST(SMResidencyTest, PartialWarpResidencyTest) {
    SM sm(0);
    SystemConfig config{1, 32, 50, 2048, 4, 65536, 65536};
    KernelResourceRequirements req{0, 0};

    ThreadBlock block = create_dummy_block(0, 50, 32);
    EXPECT_TRUE(sm.can_admit(block, config, req));
    sm.allocate_block(block, config, req);

    EXPECT_EQ(sm.get_allocated_blocks(), 1);
    EXPECT_EQ(sm.get_allocated_warps(), 2);
    EXPECT_EQ(sm.get_allocated_threads(), 50);
}

TEST(GPUPatchIntegrationTest, DynamicDispatchAfterCompletion) {
    GPU gpu(1);
    SystemConfig config{1, 32, 64, 2048, 2, 65536, 65536}; // Max 2 blocks per SM
    KernelResourceRequirements req{0, 0};

    Grid grid;
    for (size_t i = 0; i < 3; ++i) { // 3 blocks total
        grid.add_block(create_dummy_block(i, 64, 32));
    }

    Kernel kernel("dummy", {}); // Empty kernel to immediately complete

    gpu.launch_kernel(kernel, grid, config, req);

    auto& sms = gpu.get_sms();
    for (auto& sm : sms) {
        sm.set_scheduler(std::make_unique<RoundRobinScheduler>());
    }
    // Only mutable access is not available directly, but we don't need it. We can just run_to_completion.

    // Tick SM until block 0 and 1 finish.
    gpu.run_to_completion(kernel);

    // Should successfully complete all 3 blocks dynamically
    EXPECT_TRUE(sms[0].is_completed());
}
