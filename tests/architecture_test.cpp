#include <gtest/gtest.h>
#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/thread_block.hpp"
#include "architecture/warp.hpp"
#include "architecture/thread.hpp"

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
