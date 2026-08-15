#include <gtest/gtest.h>
#include "analysis/analysis_types.hpp"
#include "analysis/analyzer.hpp"
#include "analysis/sweep_engine.hpp"

using namespace sim_sm;

TEST(AnalysisTest, DeltaCalculations) {
    std::vector<SimulationResult> results(2);
    
    // Baseline
    results[0].run_name = "Base";
    results[0].total_cycles = 1000;
    results[0].ipc = 1.0;
    results[0].amat = 10.0;
    results[0].occupancy = 0.5;
    results[0].fairness = 0.9;
    
    // Config 1
    results[1].run_name = "Config1";
    results[1].total_cycles = 800; // 20% faster
    results[1].ipc = 1.25; // 25% higher IPC
    results[1].amat = 8.0; // 20% lower AMAT
    results[1].occupancy = 0.6; // +0.1
    results[1].fairness = 0.8; // -0.1
    
    ArchitecturalAnalyzer analyzer;
    auto analysis = analyzer.analyze(results, 0);
    
    ASSERT_EQ(analysis.size(), 2);
    EXPECT_TRUE(analysis[0].is_baseline);
    EXPECT_FALSE(analysis[1].is_baseline);
    
    EXPECT_DOUBLE_EQ(analysis[1].speedup, 1.25);
    EXPECT_DOUBLE_EQ(analysis[1].cycle_delta_pct, -20.0);
    EXPECT_DOUBLE_EQ(analysis[1].ipc_delta_pct, 25.0);
    EXPECT_DOUBLE_EQ(analysis[1].amat_delta_pct, -20.0);
    EXPECT_DOUBLE_EQ(analysis[1].occupancy_delta, 0.1);
    EXPECT_DOUBLE_EQ(analysis[1].fairness_delta, -0.1);
}

TEST(AnalysisTest, BottleneckDiagnosis) {
    std::vector<SimulationResult> results(1);
    
    // Compute Bound
    results[0].run_name = "ComputeBound";
    results[0].config.num_sms = 4;
    results[0].total_cycles = 1000;
    results[0].ipc = 2.0; 
    results[0].ipc_utilization = 0.9;
    results[0].amat = 5.0; 
    results[0].occupancy = 1.0;
    results[0].fairness = 1.0;
    results[0].l1_hit_rate = 0.95;
    
    ArchitecturalAnalyzer analyzer;
    auto analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Compute");
    
    // Memory Bound
    results[0].ipc_utilization = 0.05;
    results[0].amat = 200.0; 
    results[0].l1_hit_rate = 0.05; 
    analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Memory Latency");

    // Unclassified (No dominant bottleneck)
    results[0].ipc_utilization = 0.5; // Moderate
    results[0].amat = 1.0; // Very fast memory
    results[0].l1_hit_rate = 0.95;
    results[0].occupancy = 0.8;
    results[0].fairness = 0.9;
    results[0].starvation_events = 0;
    results[0].bank_conflicts = 0;
    results[0].bank_conflict_rate = 0.0;
    results[0].no_ready_warp_cycles = 1000 * 4 * 0.15; // Between 0.1 and 0.2 to avoid compute and memory/occupancy pressure
    analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Unclassified");
    EXPECT_EQ(analysis[0].confidence, "Low");
    
    // Occupancy Bound
    results[0].occupancy = 0.1; // Very low
    results[0].no_ready_warp_cycles = 1000 * 4 * 0.5; // High NoReadyWarpCycles without memory pressure
    analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Occupancy");

    // Scheduler Inefficiency
    results[0].occupancy = 1.0;
    results[0].no_ready_warp_cycles = 0;
    results[0].starvation_events = 100;
    analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Scheduler Inefficiency");
    
    // Bank Conflicts
    results[0].starvation_events = 0;
    results[0].bank_conflict_rate = 0.5;
    results[0].bank_conflicts = 1000 * 4 * 0.1; // Significant cost
    analysis = analyzer.analyze(results, 0);
    EXPECT_EQ(analysis[0].primary_bottleneck, "Bank Conflicts");
    
    // Secondary bottlenecks
    results[0].starvation_events = 100; // Also has scheduler issue
    analysis = analyzer.analyze(results, 0);
    // Either Scheduler or Bank Conflicts could be primary depending on score ties, but both should be present
    bool has_scheduler = (analysis[0].primary_bottleneck == "Scheduler Inefficiency" || 
                          std::find(analysis[0].secondary_bottlenecks.begin(), analysis[0].secondary_bottlenecks.end(), "Scheduler Inefficiency") != analysis[0].secondary_bottlenecks.end());
    bool has_bank = (analysis[0].primary_bottleneck == "Bank Conflicts" || 
                     std::find(analysis[0].secondary_bottlenecks.begin(), analysis[0].secondary_bottlenecks.end(), "Bank Conflicts") != analysis[0].secondary_bottlenecks.end());
    EXPECT_TRUE(has_scheduler);
    EXPECT_TRUE(has_bank);
}

TEST(AnalysisTest, SweepEvidenceIsolation) {
    std::vector<SimulationResult> results(2);
    results[0].config.num_sms = 1;
    results[0].config.scheduler_policy = "RR";
    results[0].config.l1_sets = 4;
    results[0].occupancy = 0.4;
    results[0].total_cycles = 10000;
    results[0].no_ready_warp_cycles = 10000 * 1 * 0.5; // Ensure occupancy pressure
    results[0].l1_hit_rate = 0.9; // Prevent memory pressure
    results[0].amat = 5.0;        // Prevent memory pressure
    
    results[1].config.num_sms = 1;
    results[1].config.scheduler_policy = "GTO"; // DIFFERENT scheduler
    results[1].config.l1_sets = 16;             // DIFFERENT L1
    results[1].config.max_warps_per_sm = 128;   // Occupancy driving change
    results[1].occupancy = 0.5;
    results[1].total_cycles = 8000;
    
    ArchitecturalAnalyzer analyzer;
    auto analysis = analyzer.analyze(results, 0);
    
    // Occupancy should NOT be validated because other parameters changed
    EXPECT_EQ(analysis[0].primary_bottleneck, "Occupancy");
    EXPECT_EQ(analysis[0].confidence, "Low"); // Downgraded
    
    // Now make them identical except for the occupancy-driving parameter (mocking a sweep)
    results[1].config.scheduler_policy = "RR";
    results[1].config.l1_sets = 4;
    analysis = analyzer.analyze(results, 0);
    
    EXPECT_EQ(analysis[0].confidence, "High"); // Validated
}

TEST(SweepEngineTest, CartesianProduct) {
    SweepEngine sweeper;
    SystemConfig base_config;
    base_config.max_registers_per_sm = 65536;
    base_config.max_shared_memory_per_sm = 65536;
    base_config.max_warps_per_sm = 64;
    
    ParameterSpace space;
    space.scheduler_policies = {"RR", "GTO", "OldestFirst"};
    space.l1_sets = {4, 8, 16};
    
    auto configs = sweeper.generate_configurations(base_config, space);
    
    EXPECT_EQ(configs.size(), 9);
    
    for (const auto& scheduler : space.scheduler_policies) {
        for (const auto& sets : space.l1_sets) {
            EXPECT_TRUE(std::any_of(
                configs.begin(), configs.end(),
                [&](const SystemConfig& c) {
                    return c.scheduler_policy == scheduler &&
                           c.l1_sets == sets;
                }));
        }
    }
}
