#pragma once

#include "runtime/config.hpp"
#include <string>
#include <vector>


namespace sim_sm {

struct ParameterSpace {
    std::vector<std::string> scheduler_policies;
    std::vector<size_t> l1_sets;
    std::vector<size_t> max_registers_per_sm;
    std::vector<size_t> max_shared_memory_per_sm;
    std::vector<size_t> max_warps_per_sm;
};

struct SimulationResult {
    SystemConfig config;
    std::string run_name;
    
    // Raw Metrics
    size_t total_cycles;
    size_t instructions_retired;
    size_t memory_instructions;
    size_t memory_transactions;
    size_t l1_hits;
    size_t l1_misses;
    size_t l2_hits;
    size_t l2_misses;
    size_t bank_conflicts;
    size_t dirty_evictions;
    size_t no_ready_warp_cycles;
    size_t max_warp_wait_cycles;
    size_t starvation_events;
    
    // Core Derived Metrics
    double ipc;
    double amat;
    double occupancy; // e.g. resident warps / max warps
    double fairness;
    double l1_hit_rate;
    double l2_hit_rate;
    double bank_conflict_rate;
    double ipc_utilization;
};

struct AnalysisResult {
    SimulationResult sim_result;
    bool is_baseline;
    
    // Baseline-relative Deltas
    double speedup; // > 1.0 means faster than baseline
    double cycle_delta_pct;
    double ipc_delta_pct;
    double amat_delta_pct;
    double occupancy_delta;
    double fairness_delta;

    // Bottleneck Diagnosis
    std::string primary_bottleneck;
    std::vector<std::string> secondary_bottlenecks;
    std::vector<std::string> contributing_factors;
    std::string confidence; // "High", "Medium", "Low"
};

} // namespace sim_sm
