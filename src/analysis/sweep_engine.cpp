#include "analysis/sweep_engine.hpp"
#include "architecture/occupancy.hpp"
#include <iostream>

namespace sim_sm {

std::vector<SystemConfig> SweepEngine::generate_configurations(const SystemConfig& base_config, const ParameterSpace& space) {
    std::vector<SystemConfig> configs;
    configs.push_back(base_config);

    // Iteratively expand the Cartesian product
    if (!space.scheduler_policies.empty()) {
        std::vector<SystemConfig> expanded;
        for (const auto& c : configs) {
            for (const auto& pol : space.scheduler_policies) {
                SystemConfig new_c = c;
                new_c.scheduler_policy = pol;
                expanded.push_back(new_c);
            }
        }
        configs = std::move(expanded);
    }

    if (!space.l1_sets.empty()) {
        std::vector<SystemConfig> expanded;
        for (const auto& c : configs) {
            for (const auto& sets : space.l1_sets) {
                SystemConfig new_c = c;
                new_c.l1_sets = sets;
                expanded.push_back(new_c);
            }
        }
        configs = std::move(expanded);
    }

    if (!space.max_registers_per_sm.empty()) {
        std::vector<SystemConfig> expanded;
        for (const auto& c : configs) {
            for (const auto& regs : space.max_registers_per_sm) {
                SystemConfig new_c = c;
                new_c.max_registers_per_sm = regs;
                expanded.push_back(new_c);
            }
        }
        configs = std::move(expanded);
    }

    if (!space.max_shared_memory_per_sm.empty()) {
        std::vector<SystemConfig> expanded;
        for (const auto& c : configs) {
            for (const auto& shmem : space.max_shared_memory_per_sm) {
                SystemConfig new_c = c;
                new_c.max_shared_memory_per_sm = shmem;
                expanded.push_back(new_c);
            }
        }
        configs = std::move(expanded);
    }
    
    if (!space.max_warps_per_sm.empty()) {
        std::vector<SystemConfig> expanded;
        for (const auto& c : configs) {
            for (const auto& warps : space.max_warps_per_sm) {
                SystemConfig new_c = c;
                new_c.max_warps_per_sm = warps;
                expanded.push_back(new_c);
            }
        }
        configs = std::move(expanded);
    }

    return configs;
}

SimulationResult SweepEngine::execute_configuration(const SystemConfig& config, const Kernel& kernel, const Grid& grid, const KernelResourceRequirements& req, size_t config_idx) {
    GPU gpu(config);
    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    SimulationResult res;
    res.config = config;
    res.run_name = "Config_" + std::to_string(config_idx);

    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;
    size_t l1_hits = 0, l1_misses = 0;
    size_t dirty_evicts = 0, bank_conflicts = 0;
    size_t no_ready_warp_cycles = 0;
    size_t max_warp_wait = 0;
    size_t starvation = 0;
    double fairness = 0.0;
    
    size_t max_cycles_acc = 0;

    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles();
        max_cycles_acc = std::max(max_cycles_acc, c.get_cycles());
        
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
        dirty_evicts += c.get_dirty_eviction_writebacks();
        bank_conflicts += c.get_bank_conflict_stalls();
        const auto& l1_stats = sm.get_l1_cache().stats();
        l1_hits += l1_stats.hits;
        l1_misses += l1_stats.misses;

        no_ready_warp_cycles += c.get_stalls(StallReason::NoReadyWarp);
        max_warp_wait = std::max(max_warp_wait, c.get_max_warp_wait_cycles());
        starvation += c.get_starvation_events();
        fairness += c.get_jains_fairness_index();
    }
    
    size_t l2_hits = gpu.get_l2_cache().stats().hits;
    size_t l2_misses = gpu.get_l2_cache().stats().misses;
    
    res.total_cycles = max_cycles_acc;
    res.instructions_retired = total_insts;
    res.memory_instructions = total_mem_insts;
    res.memory_transactions = total_mem_txns;
    res.l1_hits = l1_hits;
    res.l1_misses = l1_misses;
    res.l2_hits = l2_hits;
    res.l2_misses = l2_misses;
    res.bank_conflicts = bank_conflicts;
    res.dirty_evictions = dirty_evicts;
    res.no_ready_warp_cycles = no_ready_warp_cycles;
    res.max_warp_wait_cycles = max_warp_wait;
    res.starvation_events = starvation;

    // Derived Metrics
    res.ipc = (max_cycles_acc > 0) ? (double)total_insts / max_cycles_acc : 0.0;
    
    double l1_miss_rate = (l1_hits + l1_misses > 0) ? (double)l1_misses / (l1_hits + l1_misses) : 0.0;
    double l2_miss_rate = (l2_hits + l2_misses > 0) ? (double)l2_misses / (l2_hits + l2_misses) : 0.0;
    res.amat = config.l1_latency + (l1_miss_rate * (config.l2_latency + l2_miss_rate * config.global_memory_latency));
    if (l1_hits + l1_misses > 0) {
        res.amat += ((double)dirty_evicts / (l1_hits + l1_misses) * config.writeback_latency);
    }

    OccupancyResult occ = OccupancyCalculator::compute(config, req);
    res.occupancy = (config.max_blocks_per_sm > 0) ? (double)occ.resident_blocks / config.max_blocks_per_sm : 0.0;
    
    size_t num_sms = gpu.get_sms().size();
    res.fairness = (num_sms > 0) ? fairness / num_sms : 0.0;
    res.l1_hit_rate = (l1_hits + l1_misses > 0) ? (double)l1_hits / (l1_hits + l1_misses) : 0.0;
    res.l2_hit_rate = (l2_hits + l2_misses > 0) ? (double)l2_hits / (l2_hits + l2_misses) : 0.0;
    res.bank_conflict_rate = (total_mem_insts > 0) ? (double)bank_conflicts / total_mem_insts : 0.0;
    
    // Theoretical max IPC is usually roughly equal to the number of SMs if each issues 1 instr/cycle
    // For now we compute utilization relative to Num SMs (assuming 1 scheduler per SM, 1 instruction per cycle).
    double theoretical_peak_ipc = static_cast<double>(config.num_sms);
    res.ipc_utilization = (theoretical_peak_ipc > 0) ? (res.ipc / theoretical_peak_ipc) : 0.0;

    return res;
}

std::vector<SimulationResult> SweepEngine::run_sweep(const SystemConfig& base_config,
                                                     const Kernel& kernel,
                                                     const Grid& grid,
                                                     const KernelResourceRequirements& req,
                                                     const ParameterSpace& space) {
    std::vector<SystemConfig> configs = generate_configurations(base_config, space);
    std::vector<SimulationResult> results;
    results.reserve(configs.size());

    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "Running configuration " << i+1 << "/" << configs.size() << "...\n";
        results.push_back(execute_configuration(configs[i], kernel, grid, req, i));
    }

    return results;
}

} // namespace sim_sm
