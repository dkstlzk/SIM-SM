#include "analysis/analyzer.hpp"
#include <algorithm>


namespace sim_sm {

std::vector<AnalysisResult> ArchitecturalAnalyzer::analyze(const std::vector<SimulationResult>& results, size_t baseline_index) {
    std::vector<AnalysisResult> analysis_results;
    if (results.empty()) return analysis_results;

    if (baseline_index >= results.size()) {
        baseline_index = 0;
    }

    const SimulationResult& baseline = results[baseline_index];

    // First pass: compute deltas and initial heuristics
    for (size_t i = 0; i < results.size(); ++i) {
        analysis_results.push_back(compute_deltas_and_bottlenecks(baseline, results[i], i == baseline_index));
    }

    // Second pass: Cross-configuration sweep validation (e.g. Occupancy evidence)
    for (size_t i = 0; i < analysis_results.size(); ++i) {
        auto& current_res = analysis_results[i];
        
        // If initial heuristic leans towards Occupancy, look for sweep evidence
        bool found_occupancy_sweep_evidence = false;
        for (size_t j = 0; j < results.size(); ++j) {
            if (i == j) continue;
            bool sched_same = (results[j].config.scheduler_policy == results[i].config.scheduler_policy);
            bool l1_same = (results[j].config.l1_sets == results[i].config.l1_sets);
            bool reg_same = (results[j].config.max_registers_per_sm == results[i].config.max_registers_per_sm);
            bool smem_same = (results[j].config.max_shared_memory_per_sm == results[i].config.max_shared_memory_per_sm);
            bool warp_same = (results[j].config.max_warps_per_sm == results[i].config.max_warps_per_sm);

            int diff_count = (!sched_same) + (!l1_same) + (!reg_same) + (!smem_same) + (!warp_same);

            // If exactly one parameter changed, and it's an occupancy-related knob, and cycles improved:
            if (diff_count == 1 && (!reg_same || !smem_same || !warp_same) &&
                results[j].occupancy > results[i].occupancy && 
                results[j].total_cycles < results[i].total_cycles) {
                found_occupancy_sweep_evidence = true;
                break;
            }
        }

        if (found_occupancy_sweep_evidence && current_res.primary_bottleneck == "Occupancy") {
            current_res.confidence = "High";
            current_res.contributing_factors.push_back("Sweep evidence validates occupancy limitation: higher occupancy config reduced cycles.");
        } else if (current_res.primary_bottleneck == "Occupancy" && current_res.confidence == "Medium") {
            // Downgrade if no sweep evidence
            current_res.confidence = "Low";
            current_res.contributing_factors.push_back("No configuration sweep found that improved performance by increasing occupancy.");
        }
        
        // Similarly for scheduler starvation
        if (current_res.primary_bottleneck == "Scheduler Inefficiency") {
            bool sched_improved = false;
            for (size_t j = 0; j < results.size(); ++j) {
                if (results[j].config.scheduler_policy != results[i].config.scheduler_policy &&
                    results[j].config.l1_sets == results[i].config.l1_sets &&
                    results[j].config.max_registers_per_sm == results[i].config.max_registers_per_sm &&
                    results[j].config.max_shared_memory_per_sm == results[i].config.max_shared_memory_per_sm &&
                    results[j].config.max_warps_per_sm == results[i].config.max_warps_per_sm) {
                    if (results[j].total_cycles < results[i].total_cycles) {
                        sched_improved = true;
                        break;
                    }
                }
            }
            if (sched_improved) {
                current_res.contributing_factors.push_back("Sweep evidence: Alternative scheduler improved cycles.");
            }
        }
    }

    return analysis_results;
}

AnalysisResult ArchitecturalAnalyzer::compute_deltas_and_bottlenecks(const SimulationResult& baseline, const SimulationResult& target, bool is_baseline) {
    AnalysisResult res;
    res.sim_result = target;
    res.is_baseline = is_baseline;

    if (target.total_cycles > 0) {
        res.speedup = (double)baseline.total_cycles / target.total_cycles;
        res.cycle_delta_pct = ((double)target.total_cycles - baseline.total_cycles) / baseline.total_cycles * 100.0;
    } else {
        res.speedup = 1.0;
        res.cycle_delta_pct = 0.0;
    }

    if (baseline.ipc > 0) {
        res.ipc_delta_pct = (target.ipc - baseline.ipc) / baseline.ipc * 100.0;
    } else {
        res.ipc_delta_pct = 0.0;
    }

    if (baseline.amat > 0) {
        res.amat_delta_pct = (target.amat - baseline.amat) / baseline.amat * 100.0;
    } else {
        res.amat_delta_pct = 0.0;
    }

    res.occupancy_delta = target.occupancy - baseline.occupancy;
    res.fairness_delta = target.fairness - baseline.fairness;

    diagnose_bottleneck(res);

    return res;
}

void ArchitecturalAnalyzer::diagnose_bottleneck(AnalysisResult& res) {
    const auto& s = res.sim_result;
    
    double total_sm_cycles = s.total_cycles * s.config.num_sms;
    
    double memory_pressure = 0.0;
    if (s.amat > s.config.l1_latency * 1.5) memory_pressure += 1.0;
    if (s.l1_hit_rate < 0.8) memory_pressure += 1.0;
    if (s.memory_instructions > 0 && (double)s.memory_transactions / s.memory_instructions > 1.2) memory_pressure += 1.0;
    if (s.no_ready_warp_cycles > total_sm_cycles * 0.2) memory_pressure += 0.5;
    
    // SIM-SM models one warp/instruction issue per SM per cycle,
    // therefore peak system IPC = num_sms. 
    double compute_pressure = 0.0;
    if (s.ipc_utilization > 0.7) compute_pressure += 2.0;
    if (s.no_ready_warp_cycles < total_sm_cycles * 0.1) compute_pressure += 1.0;
    
    double occupancy_pressure = 0.0;
    if (s.occupancy < 0.5) occupancy_pressure += 1.0;
    if (s.no_ready_warp_cycles > total_sm_cycles * 0.2 && memory_pressure < 1.5) occupancy_pressure += 1.0;
    
    double scheduler_pressure = 0.0;
    if (s.starvation_events > 0) scheduler_pressure += 2.0;
    if (s.max_warp_wait_cycles > 500) scheduler_pressure += 1.0; // tail latency issue
    if (s.fairness < 0.7 && s.no_ready_warp_cycles > total_sm_cycles * 0.1) scheduler_pressure += 0.5;
    
    double bank_conflict_pressure = 0.0;
    if (s.bank_conflict_rate > 0.1) bank_conflict_pressure += 1.0;
    if (s.bank_conflicts > total_sm_cycles * 0.05) bank_conflict_pressure += 1.0; // High absolute stall impact
    
    struct Candidate {
        std::string name;
        double score;
        std::vector<std::string> evidence;
    };
    
    std::vector<Candidate> candidates;
    
    Candidate comp{"Compute", compute_pressure, {}};
    if (s.ipc_utilization > 0.7) comp.evidence.push_back("High IPC utilization (" + std::to_string(s.ipc_utilization) + ")");
    if (s.no_ready_warp_cycles < total_sm_cycles * 0.1) comp.evidence.push_back("Low NoReadyWarpCycles");
    candidates.push_back(comp);
    
    Candidate mem{"Memory Latency", memory_pressure, {}};
    if (s.amat > s.config.l1_latency * 1.5) mem.evidence.push_back("High AMAT (" + std::to_string(s.amat) + ")");
    if (s.l1_hit_rate < 0.8) mem.evidence.push_back("Low L1 hit rate (" + std::to_string(s.l1_hit_rate) + ")");
    if (s.memory_instructions > 0 && (double)s.memory_transactions / s.memory_instructions > 1.2) mem.evidence.push_back("Uncoalesced accesses driving high memory transactions");
    candidates.push_back(mem);
    
    Candidate occ{"Occupancy", occupancy_pressure, {}};
    if (s.occupancy < 0.5) occ.evidence.push_back("Low occupancy (" + std::to_string(s.occupancy) + ")");
    if (s.no_ready_warp_cycles > total_sm_cycles * 0.2) occ.evidence.push_back("High NoReadyWarpCycles without severe memory pressure");
    candidates.push_back(occ);
    
    Candidate sched{"Scheduler Inefficiency", scheduler_pressure, {}};
    if (s.starvation_events > 0) sched.evidence.push_back("Starvation events detected (" + std::to_string(s.starvation_events) + ")");
    if (s.max_warp_wait_cycles > 500) sched.evidence.push_back("High max warp wait cycles (" + std::to_string(s.max_warp_wait_cycles) + ")");
    if (s.fairness < 0.7) sched.evidence.push_back("Low fairness (" + std::to_string(s.fairness) + ")");
    candidates.push_back(sched);
    
    Candidate bc{"Bank Conflicts", bank_conflict_pressure, {}};
    if (s.bank_conflict_rate > 0.1) bc.evidence.push_back("High bank conflict rate (" + std::to_string(s.bank_conflict_rate) + ")");
    if (s.bank_conflicts > s.total_cycles * 0.05) bc.evidence.push_back("Bank conflicts represent significant cycle cost");
    candidates.push_back(bc);
    
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });
    
    if (candidates[0].score < 1.0) {
        res.primary_bottleneck = "Unclassified";
        res.confidence = "Low";
        res.contributing_factors.push_back("No dominant architectural bottleneck identified.");
        return;
    }
    
    res.primary_bottleneck = candidates[0].name;
    res.contributing_factors = candidates[0].evidence;
    
    if (candidates[0].score >= 2.5) res.confidence = "High";
    else if (candidates[0].score >= 1.5) res.confidence = "Medium";
    else res.confidence = "Low";
    
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].score >= 1.0) {
            res.secondary_bottlenecks.push_back(candidates[i].name);
            for (const auto& ev : candidates[i].evidence) {
                res.contributing_factors.push_back(candidates[i].name + " -> " + ev);
            }
        }
    }
    
    if (res.contributing_factors.empty()) {
        res.contributing_factors.push_back("No strong evidence collected.");
    }
}

} // namespace sim_sm
