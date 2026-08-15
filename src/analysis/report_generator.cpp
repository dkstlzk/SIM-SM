#include "analysis/report_generator.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>

namespace sim_sm {

std::string ReportGenerator::generate_markdown_report(const std::vector<AnalysisResult>& results, const std::string& title) {
    if (results.empty()) return "# " + title + "\n\nNo results provided.\n";

    std::stringstream ss;
    ss << "# " << title << "\n\n";
    
    // Find best config
    size_t best_idx = 0;
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].speedup > results[best_idx].speedup) {
            best_idx = i;
        }
    }
    const auto& best = results[best_idx];

    // Executive Summary
    auto baseline_it = std::find_if(results.begin(), results.end(), [](const AnalysisResult& r) { return r.is_baseline; });
    const auto& baseline = (baseline_it != results.end()) ? *baseline_it : results[0];

    ss << "## Executive Summary\n\n";
    ss << "**Baseline configuration:** " << baseline.sim_result.run_name << "\n";
    ss << "- Scheduler: " << baseline.sim_result.config.scheduler_policy << "\n";
    ss << "- L1 Sets: " << baseline.sim_result.config.l1_sets << "\n\n";

    ss << "**Best configuration:** " << best.sim_result.run_name << "\n";
    ss << "- Scheduler: " << best.sim_result.config.scheduler_policy << "\n";
    ss << "- L1 Sets: " << best.sim_result.config.l1_sets << "\n";
    ss << "- Max Registers: " << best.sim_result.config.max_registers_per_sm << "\n";
    ss << "- Max Shared Memory: " << best.sim_result.config.max_shared_memory_per_sm << "\n";
    ss << "- Max Warps: " << best.sim_result.config.max_warps_per_sm << "\n\n";
    
    ss << "**Speedup over baseline:** " << std::fixed << std::setprecision(2) << best.speedup << "x\n\n";
    ss << "**Primary bottleneck (Best Config):** " << best.primary_bottleneck << "\n\n";
    ss << "**Confidence:** " << best.confidence << "\n\n";

    // Configuration Comparison
    ss << "## Configuration Comparison\n\n";
    ss << "| Config | Scheduler | L1 Sets | Cycles | IPC | AMAT | Occupancy | Fairness | Speedup |\n";
    ss << "|--------|-----------|---------|--------|-----|------|-----------|----------|---------|\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "| " << r.sim_result.run_name << (r.is_baseline ? " (Base)" : "") << " "
           << "| " << r.sim_result.config.scheduler_policy << " "
           << "| " << r.sim_result.config.l1_sets << " "
           << "| " << r.sim_result.total_cycles << " "
           << "| " << std::fixed << std::setprecision(2) << r.sim_result.ipc << " "
           << "| " << std::fixed << std::setprecision(2) << r.sim_result.amat << " "
           << "| " << std::fixed << std::setprecision(2) << r.sim_result.occupancy << " "
           << "| " << std::fixed << std::setprecision(2) << r.sim_result.fairness << " "
           << "| " << std::fixed << std::setprecision(2) << r.speedup << "x |\n";
    }
    ss << "\n";
    
    // Bottleneck Diagnosis
    ss << "## Bottleneck Diagnosis\n\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "### " << r.sim_result.run_name << (r.is_baseline ? " (Baseline)" : "") << "\n";
        ss << "**Primary:** " << r.primary_bottleneck << "\n";
        if (!r.secondary_bottlenecks.empty()) {
            ss << "**Secondary:** ";
            for (size_t j = 0; j < r.secondary_bottlenecks.size(); ++j) {
                ss << r.secondary_bottlenecks[j] << (j + 1 < r.secondary_bottlenecks.size() ? ", " : "");
            }
            ss << "\n";
        }
        ss << "**Evidence:**\n";
        for (const auto& ev : r.contributing_factors) {
            ss << "- " << ev << "\n";
        }
        
        if (!r.is_baseline) {
            ss << "**Deltas vs Baseline:**\n";
            ss << "- Cycles: " << std::showpos << std::fixed << std::setprecision(1) << r.cycle_delta_pct << "%\n";
            ss << "- IPC: " << std::showpos << r.ipc_delta_pct << "%\n";
            ss << "- AMAT: " << std::showpos << r.amat_delta_pct << "%\n";
            ss << "- Occupancy: " << std::showpos << r.occupancy_delta << "\n";
            ss << "- Fairness: " << std::showpos << r.fairness_delta << "\n";
            ss << std::noshowpos; // reset
        }
        ss << "\n";
    }
    
    // Parameter Sensitivity
    ss << "## Parameter Sensitivity\n\n";
    
    std::map<std::string, std::vector<double>> sched_speedups;
    std::map<size_t, std::vector<double>> l1_speedups;
    for (const auto& r : results) {
        sched_speedups[r.sim_result.config.scheduler_policy].push_back(r.speedup);
        l1_speedups[r.sim_result.config.l1_sets].push_back(r.speedup);
    }
    
    ss << "### Average Speedup by Scheduler Policy\n";
    for (const auto& [policy, speeds] : sched_speedups) {
        double avg = 0; for(auto v: speeds) avg += v; avg /= speeds.size();
        ss << "- **" << policy << "**: " << std::fixed << std::setprecision(2) << avg << "x\n";
    }
    
    ss << "\n### Average Speedup by L1 Sets\n";
    for (const auto& [sets, speeds] : l1_speedups) {
        double avg = 0; for(auto v: speeds) avg += v; avg /= speeds.size();
        ss << "- **" << sets << " sets**: " << std::fixed << std::setprecision(2) << avg << "x\n";
    }
    ss << "\n*Note: These averages are descriptive sensitivity summaries across tested configurations and may include interactions between simultaneously swept parameters. Causal bottleneck conclusions are based exclusively on matched-configuration isolation.*\n\n";
    
    // Architectural Recommendations (simple synthesis)
    ss << "## Architectural Recommendations\n\n";
    
    if (best.primary_bottleneck == "Memory Latency") {
        ss << "1. The best configuration is still memory latency bound. Consider increasing `l1_sets`, `l2_sets`, or changing cache policies.\n";
        ss << "   *(Future capability note: hardware prefetching is not currently supported in SIM-SM but would further mitigate this.)*\n";
    } else if (best.primary_bottleneck == "Compute") {
        ss << "1. The best configuration successfully achieves high IPC and is compute bound. Expanding SM execution units (issue width) would be required for further scaling.\n";
        ss << "   *(Future capability note: SIM-SM currently assumes peak IPC = 1 instruction per SM per cycle.)*\n";
    } else if (best.primary_bottleneck == "Occupancy") {
        ss << "1. Performance is currently limited by the number of resident warps. Increase `max_registers_per_sm`, `max_shared_memory_per_sm`, or `max_warps_per_sm`.\n";
    } else if (best.primary_bottleneck == "Unclassified") {
        ss << "1. No single dominant architectural bottleneck was found. The system is well-balanced or limited by instruction mix.\n";
    }
    
    if (baseline.primary_bottleneck == "Scheduler Inefficiency" && best.primary_bottleneck != "Scheduler Inefficiency") {
        ss << "2. Changing the scheduler policy successfully mitigated initial scheduler-related stalls.\n";
    }
    
    return ss.str();
}

} // namespace sim_sm
