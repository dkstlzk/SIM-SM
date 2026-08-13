#include <gtest/gtest.h>
#include "runtime/benchmarks.hpp"
#include "architecture/gpu.hpp"
#include "scheduling/round_robin_scheduler.hpp"

using namespace sim_sm;

// We need to test the memcpy kernel in isolation using CPU reference
TEST(BenchmarkTest, MemcpyCorrectnessContiguous) {
    SystemConfig config;
    config.num_sms = 1;
    config.block_size = 64;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;
    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    size_t num_threads = 64;
    std::string pattern = "contiguous";
    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    
    std::vector<int> cpu_src(num_threads);
    std::vector<size_t> offsets(num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t offset = i * 4;
        offsets[i] = offset;
        int val = (i % 255) + 1;
        cpu_src[i] = val;
        gm.store(0 + offset, val);
        gm.store(1000000 + offset, 0);
    }

    int iterations = 10;
    sim_sm::Kernel kernel = generate_memcpy_kernel(iterations);
    sim_sm::Grid grid = build_memcpy_grid(num_threads, config.block_size, config.warp_size, pattern);

    sim_sm::KernelResourceRequirements req = {10, 0};

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t offset = offsets[i];
        int gpu_val = gm.load(1000000 + offset);
        EXPECT_EQ(gpu_val, cpu_src[i]) << "Mismatch at thread " << i;
    }
}

TEST(BenchmarkTest, MemcpyCorrectnessStrided) {
    SystemConfig config;
    config.num_sms = 1;
    config.block_size = 64;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;
    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    size_t num_threads = 64;
    std::string pattern = "strided";
    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    
    std::vector<int> cpu_src(num_threads);
    std::vector<size_t> offsets(num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t offset = i * 32;
        offsets[i] = offset;
        int val = (i % 255) + 1;
        cpu_src[i] = val;
        gm.store(0 + offset, val);
        gm.store(1000000 + offset, 0);
    }

    int iterations = 10;
    sim_sm::Kernel kernel = generate_memcpy_kernel(iterations);
    sim_sm::Grid grid = build_memcpy_grid(num_threads, config.block_size, config.warp_size, pattern);

    sim_sm::KernelResourceRequirements req = {10, 0};

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t offset = offsets[i];
        int gpu_val = gm.load(1000000 + offset);
        EXPECT_EQ(gpu_val, cpu_src[i]) << "Mismatch at thread " << i;
    }
}

TEST(BenchmarkTest, ReductionCorrectness) {
    SystemConfig config;
    config.num_sms = 1;
    config.block_size = 64;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;

    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    size_t num_elements = 64;
    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    
    int expected_sum = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        int val = (i % 5) + 1;
        expected_sum += val;
        gm.store(i * 4, val);
    }
    gm.store(num_elements * 4, 0); // out

    sim_sm::Kernel kernel = generate_reduction_kernel(config.block_size);
    sim_sm::Grid grid = build_reduction_grid(num_elements, config.block_size, config.warp_size);

    sim_sm::KernelResourceRequirements req = {32, config.block_size * 4};

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    int gpu_val = gm.load(num_elements * 4);
    EXPECT_EQ(gpu_val, expected_sum);

    size_t total_barrier_stalls = 0;
    for (const auto& sm : gpu.get_sms()) {
        total_barrier_stalls += sm.get_counters().get_warp_barrier_stall_cycles();
    }
    EXPECT_GT(total_barrier_stalls, 0);
}

TEST(BenchmarkTest, HistogramWriteConflicts) {
    SystemConfig config;
    config.num_sms = 1;
    config.block_size = 64;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;

    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    size_t num_elements = 64;
    size_t num_bins = 16;
    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    
    // Force 100% collision in each warp (all write to bin 0)
    for (size_t i = 0; i < num_elements; ++i) {
        gm.store(i * 4, 0); 
    }
    
    // Output bins zeroed
    for (size_t i = 0; i < num_bins; ++i) {
        gm.store(num_elements * 4 + i * 4, 0); 
    }

    sim_sm::Kernel kernel = generate_histogram_kernel(num_bins, config.block_size);
    sim_sm::Grid grid = build_histogram_grid(num_elements, config.block_size, config.warp_size, num_bins);
    sim_sm::KernelResourceRequirements req = {32, num_bins * 4};

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    size_t conflicts = gpu.get_sms()[0].get_counters().get_stalls(sim_sm::StallReason::WriteConflict);
    EXPECT_EQ(conflicts, 62);

    // Also check correctness of the histogram
    EXPECT_EQ(gm.load(num_elements * 4), 64); // Bin 0 should have 64 elements
}

TEST(BenchmarkTest, AtomicAddSameAddress) {
    sim_sm::SystemConfig config;
    config.num_sms = 1;
    config.block_size = 32;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;

    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    gm.store(100000, 0); // target address (>= 65536 for global memory)

    std::vector<sim_sm::Instruction> insts;
    insts.push_back({sim_sm::Opcode::MOV, 0, -1, -1, 100000}); // addr = 100000
    insts.push_back({sim_sm::Opcode::MOV, 1, -1, -1, 1}); // val = 1
    insts.push_back({sim_sm::Opcode::ATOMIC_ADD, -1, 1, 0, 0}); // ATOMIC_ADD val to addr

    sim_sm::Kernel kernel("atomic_same", insts);
    sim_sm::Grid grid;
    sim_sm::ThreadBlock block(0);
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        warp.add_thread(t);
    }
    block.add_warp(warp);
    grid.add_block(block);

    sim_sm::KernelResourceRequirements req = {10, 0};
    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    EXPECT_EQ(gm.load(100000), 32);

    size_t conflicts = gpu.get_sms()[0].get_counters().get_stalls(sim_sm::StallReason::WriteConflict);
    EXPECT_EQ(conflicts, 31);
}

TEST(BenchmarkTest, AtomicAddDistinctAddresses) {
    sim_sm::SystemConfig config;
    config.num_sms = 1;
    config.block_size = 32;
    config.warp_size = 32;
    config.max_blocks_per_sm = 16;
    config.max_threads_per_sm = 1024;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;

    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 10485760);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    for(int i = 0; i < 32; ++i) gm.store(100000 + i * 4, 0);

    std::vector<sim_sm::Instruction> insts;
    insts.push_back({sim_sm::Opcode::MOV, 1, -1, -1, 1}); // val = 1
    insts.push_back({sim_sm::Opcode::MOV, 3, -1, -1, 4}); 
    insts.push_back({sim_sm::Opcode::MUL, 4, 2, 3, 0});   // R4 = R2 * 4
    insts.push_back({sim_sm::Opcode::MOV, 5, -1, -1, 100000});
    insts.push_back({sim_sm::Opcode::ADD, 6, 5, 4, 0});   // R6 = 100000 + R2*4
    insts.push_back({sim_sm::Opcode::ATOMIC_ADD, -1, 1, 6, 0}); 

    sim_sm::Kernel kernel("atomic_distinct", insts);
    sim_sm::Grid grid;
    sim_sm::ThreadBlock block(0);
    sim_sm::Warp warp(0);
    for (int i = 0; i < 32; ++i) {
        sim_sm::Thread t(i, 0, i, 0, i);
        t.registers().write(2, i); // R2 = thread ID
        warp.add_thread(t);
    }
    block.add_warp(warp);
    grid.add_block(block);

    sim_sm::KernelResourceRequirements req = {10, 0};
    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(gm.load(100000 + i * 4), 1);
    }

    size_t conflicts = gpu.get_sms()[0].get_counters().get_stalls(sim_sm::StallReason::WriteConflict);
    EXPECT_EQ(conflicts, 0);
}
