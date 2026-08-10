#include <gtest/gtest.h>
#include "memory/cache.hpp"
#include "runtime/trace_logger.hpp"
#include "architecture/gpu.hpp"
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace sim_sm;

struct PerfMetrics {
    size_t cycles;
    size_t insts;
    size_t mem_txns;
};

PerfMetrics run_trace_workload(TraceLevel level, const std::string& trace_file) {
    GPU gpu(1, 1, 1, 1, 1, 32, 1024, "LRU");
    gpu.get_sms()[0].set_scheduler(std::make_unique<RoundRobinScheduler>());
    
    std::unique_ptr<TraceLogger> logger;
    if (level != TraceLevel::None) {
        logger = std::make_unique<TraceLogger>(level, trace_file);
        gpu.set_trace_logger(logger.get());
    }
    
    std::vector<Instruction> insts = {
        {Opcode::LOAD, 1, 0, -1, 0}, // GLOBAL
        {Opcode::MOV, 2, -1, -1, 0x10000000},
        {Opcode::STORE, -1, 1, 2, 4}, // SHARED
        {Opcode::ADD, 3, 1, 1, 0}
    };
    Kernel kernel("trace_test_kernel", insts);
    Grid grid;
    ThreadBlock block(0);
    Warp warp(0);
    warp.add_thread(Thread(0, 0, 0, 0, 0));
    block.add_warp(warp);
    grid.add_block(block);
    
    SystemConfig config;
    config.num_sms = 1;
    config.warp_size = 1;
    config.block_size = 1;
    config.max_threads_per_sm = 1024;
    config.max_blocks_per_sm = 8;
    config.max_shared_memory_per_sm = 65536;
    config.max_registers_per_sm = 65536;
    
    KernelResourceRequirements req = { 4, 0 };
    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    const auto& c = gpu.get_sms()[0].get_counters();
    return {c.get_cycles(), c.get_instructions_retired(), c.get_memory_transactions()};
}

TEST(TraceTest, TracingWritesToFileAndHasMemorySpaces) {
    std::string trace_file = "test_trace_all.txt";
    if (std::filesystem::exists(trace_file)) std::filesystem::remove(trace_file);
    
    run_trace_workload(TraceLevel::All, trace_file);
    
    ASSERT_TRUE(std::filesystem::exists(trace_file));
    std::ifstream in(trace_file);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();
    
    EXPECT_TRUE(content.find("Scheduler:") != std::string::npos) << "Actual content:\n" << content;
    EXPECT_TRUE(content.find("Opcode=LOAD addr=0x0") != std::string::npos) << "Actual content:\n" << content;
    EXPECT_TRUE(content.find("Opcode=STORE addr=0x4") != std::string::npos) << "Actual content:\n" << content;
    EXPECT_TRUE(content.find("MemorySpace=GLOBAL") != std::string::npos) << "Actual content:\n" << content;
    EXPECT_TRUE(content.find("MemorySpace=SHARED") != std::string::npos) << "Actual content:\n" << content;
    EXPECT_TRUE(content.find("Result=MISS") != std::string::npos || content.find("Result=HIT") != std::string::npos) << "Actual content:\n" << content;
    
    std::filesystem::remove(trace_file);
}

TEST(TraceTest, TracingLevelsIsolated) {
    std::string trace_sched = "test_trace_sched.txt";
    std::string trace_mem = "test_trace_mem.txt";
    std::string trace_cache = "test_trace_cache.txt";
    
    run_trace_workload(TraceLevel::Scheduler, trace_sched);
    run_trace_workload(TraceLevel::Memory, trace_mem);
    run_trace_workload(TraceLevel::Cache, trace_cache);
    
    auto read_file = [](const std::string& path) {
        std::ifstream in(path);
        std::stringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    };
    
    std::string content_sched = read_file(trace_sched);
    std::string content_mem = read_file(trace_mem);
    std::string content_cache = read_file(trace_cache);
    
    EXPECT_TRUE(content_sched.find("Scheduler:") != std::string::npos) << "content_sched:\n" << content_sched;
    EXPECT_TRUE(content_sched.find("Memory Event:") == std::string::npos) << "content_sched:\n" << content_sched;
    EXPECT_TRUE(content_sched.find("Cache Event:") == std::string::npos) << "content_sched:\n" << content_sched;
    
    EXPECT_TRUE(content_mem.find("Memory Event:") != std::string::npos) << "content_mem:\n" << content_mem;
    EXPECT_TRUE(content_mem.find("Scheduler:") == std::string::npos) << "content_mem:\n" << content_mem;
    EXPECT_TRUE(content_mem.find("Cache Event:") == std::string::npos) << "content_mem:\n" << content_mem;
    
    EXPECT_TRUE(content_cache.find("Cache Event:") != std::string::npos) << "content_cache:\n" << content_cache;
    EXPECT_TRUE(content_cache.find("Scheduler:") == std::string::npos) << "content_cache:\n" << content_cache;
    EXPECT_TRUE(content_cache.find("Memory Event:") == std::string::npos) << "content_cache:\n" << content_cache;
    
    std::filesystem::remove(trace_sched);
    std::filesystem::remove(trace_mem);
    std::filesystem::remove(trace_cache);
}

TEST(TraceTest, TraceOffEquivalence) {
    std::string temp = "test_trace_temp.txt";
    PerfMetrics metrics_on = run_trace_workload(TraceLevel::All, temp);
    PerfMetrics metrics_off = run_trace_workload(TraceLevel::None, "");
    
    EXPECT_EQ(metrics_on.cycles, metrics_off.cycles);
    EXPECT_EQ(metrics_on.insts, metrics_off.insts);
    EXPECT_EQ(metrics_on.mem_txns, metrics_off.mem_txns);
    
    if (std::filesystem::exists(temp)) std::filesystem::remove(temp);
}
