#include "runtime/benchmarks.hpp"
#include "architecture/gpu.hpp"
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"
#include "architecture/occupancy.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include "scheduling/greedy_scheduler.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace sim_sm {

sim_sm::Kernel generate_vector_add_kernel(int iterations) {
    std::vector<sim_sm::Instruction> insts;
    for (int i = 0; i < iterations; ++i) {
        int offset = i * 4;
        insts.push_back({sim_sm::Opcode::LOAD, 3, 0, -1, offset});
        insts.push_back({sim_sm::Opcode::LOAD, 4, 1, -1, offset});
        insts.push_back({sim_sm::Opcode::ADD, 5, 3, 4, 0});
        insts.push_back({sim_sm::Opcode::STORE, 2, 5, 2, offset});
    }
    return sim_sm::Kernel("vector_add", insts);
}

sim_sm::Grid build_benchmark_grid(size_t num_threads, size_t block_size, size_t warp_size, bool strided) {
    sim_sm::Grid grid;
    size_t num_blocks = (num_threads + block_size - 1) / block_size;
    if (num_threads == 0) num_blocks = 0;
    
    for (size_t b = 0; b < num_blocks; ++b) {
        sim_sm::ThreadBlock block(b);
        size_t threads_in_this_block = std::min(block_size, num_threads - b * block_size);
        size_t warps_in_this_block = (threads_in_this_block + warp_size - 1) / warp_size;
        
        for (size_t w = 0; w < warps_in_this_block; ++w) {
            sim_sm::Warp warp(w);
            size_t threads_in_this_warp = std::min(warp_size, threads_in_this_block - w * warp_size);
            
            for (size_t l = 0; l < threads_in_this_warp; ++l) {
                size_t local_id = w * warp_size + l;
                size_t global_id = b * block_size + local_id;
                sim_sm::Thread thread(global_id, b, local_id, w, l);
                
                size_t stride = strided ? 32 : 4; 
                thread.registers().write(0, 0 + global_id * stride);
                thread.registers().write(1, 10000 + global_id * stride);
                thread.registers().write(2, 20000 + global_id * stride);
                
                warp.add_thread(thread);
            }
            block.add_warp(warp);
        }
        grid.add_block(block);
    }
    return grid;
}

void run_experiment(const std::string& name, const SystemConfig& config, 
                    const std::string& scheduler_type, bool strided,
                    size_t l1_sets = 4, size_t l1_assoc = 4) {
    
    std::filesystem::create_directories("results");
    std::ofstream out("results/" + name + ".csv");
    out << "Experiment,SMs,TotalCycles,Instructions,IPC,MemInsts,MemTxns,Occupancy,BlocksPerSM" << "\n";
    
    sim_sm::GPU gpu(config.num_sms, l1_sets, l1_assoc, 16, 8, 32, 1048576); 
    
    for (auto& sm : gpu.get_sms()) {
        if (scheduler_type == "greedy") {
            sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
        } else {
            sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
        }
    }
    
    size_t num_threads = 2048;
    sim_sm::KernelResourceRequirements req = {10, 0}; 
    
    sim_sm::Kernel kernel = generate_vector_add_kernel(100); 
    sim_sm::Grid grid = build_benchmark_grid(num_threads, config.block_size, config.warp_size, strided);
    
    OccupancyResult occ = OccupancyCalculator::compute(config, req);
    
    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);
    
    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;
    
    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles(); 
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
    }
    
    size_t max_cycles = 0;
    for (const auto& sm : gpu.get_sms()) {
        max_cycles = std::max(max_cycles, sm.get_counters().get_cycles());
    }
    
    double ipc = (max_cycles > 0) ? (double)total_insts / max_cycles : 0.0;
    
    out << name << "," << config.num_sms << "," << max_cycles << "," 
        << total_insts << "," << std::fixed << std::setprecision(2) << ipc << "," 
        << total_mem_insts << "," << total_mem_txns << ","
        << occ.occupancy_percentage << "," << occ.resident_blocks << "\n";
    out.close();
}

void run_cache_experiment(const std::string& name, const SystemConfig& config, size_t l1_sets, size_t l1_assoc) {
    std::filesystem::create_directories("results");
    std::ofstream out("results/" + name + ".csv");
    out << "Experiment,SMs,TotalCycles,Instructions,IPC,MemInsts,MemTxns,Occupancy,BlocksPerSM\n";
    
    // 1 SM to isolate cache behavior without inter-SM interference
    sim_sm::GPU gpu(1, l1_sets, l1_assoc, 16, 8, 32, 1048576); 
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }
    
    size_t num_threads = 64; 
    SystemConfig cache_config = config;
    cache_config.num_sms = 1;
    sim_sm::KernelResourceRequirements req = {10, 0}; 
    
    std::vector<sim_sm::Instruction> insts;
    for (int iter = 0; iter < 10; ++iter) {
        for (int i = 0; i < 8; ++i) {
            int offset = i * 32; 
            insts.push_back({sim_sm::Opcode::LOAD, 3, 0, -1, offset});
        }
    }
    sim_sm::Kernel kernel("cache_stress", insts); 
    sim_sm::Grid grid = build_benchmark_grid(num_threads, config.block_size, config.warp_size, false);
    
    OccupancyResult occ = OccupancyCalculator::compute(cache_config, req);
    gpu.launch_kernel(kernel, grid, cache_config, req);
    gpu.run_to_completion(kernel);
    
    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;
    
    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles(); 
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
    }
    
    double ipc = (total_cycles > 0) ? (double)total_insts / total_cycles : 0.0;
    
    out << name << "," << 1 << "," << total_cycles << "," 
        << total_insts << "," << std::fixed << std::setprecision(2) << ipc << "," 
        << total_mem_insts << "," << total_mem_txns << ","
        << occ.occupancy_percentage << "," << occ.resident_blocks << "\n";
    out.close();
}

void run_benchmarks(const std::string& config_path) {
    SystemConfig base_config = load_config(config_path);
    
    // 1. scheduler_comparison
    run_experiment("scheduler_rr", base_config, "rr", false);
    run_experiment("scheduler_greedy", base_config, "greedy", false);
    
    // 2. cache_sweep (Small L1: 2 sets 2 assoc vs Large L1: 16 sets 4 assoc)
    run_cache_experiment("cache_small", base_config, 2, 2);
    run_cache_experiment("cache_large", base_config, 16, 4);
    
    // 3. coalescing_sweep
    run_experiment("coalesced", base_config, "rr", false);
    run_experiment("strided", base_config, "rr", true);
    
    // 4. occupancy_sweep (change block size)
    SystemConfig small_block = base_config;
    small_block.block_size = 64;
    run_experiment("occupancy_64", small_block, "rr", false);
    
    SystemConfig large_block = base_config;
    large_block.block_size = 256;
    run_experiment("occupancy_256", large_block, "rr", false);
}

} // namespace sim_sm
