#pragma once

#include "analysis/analysis_types.hpp"
#include <vector>


namespace sim_sm {

class ArchitecturalAnalyzer {
public:
    // Process a set of raw SimulationResults, using the first element as the baseline,
    // or specifically indicating the baseline via index.
    std::vector<AnalysisResult> analyze(const std::vector<SimulationResult>& results, size_t baseline_index = 0);

private:
    AnalysisResult compute_deltas_and_bottlenecks(const SimulationResult& baseline, const SimulationResult& target, bool is_baseline);
    
    // Bottleneck diagnosis logic
    void diagnose_bottleneck(AnalysisResult& res);
};

} // namespace sim_sm
