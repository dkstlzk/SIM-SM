#pragma once

#include "analysis/analysis_types.hpp"
#include "architecture/gpu.hpp"
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"

namespace sim_sm {

class SweepEngine {
public:
    // Runs a Cartesian product of all parameters in the ParameterSpace against the given kernel.
    // The base_config provides the default values for parameters not being swept.
    // Returns a vector of raw SimulationResults.
    std::vector<SimulationResult> run_sweep(const SystemConfig& base_config,
                                            const Kernel& kernel,
                                            const Grid& grid,
                                            const KernelResourceRequirements& req,
                                            const ParameterSpace& space);

    std::vector<SystemConfig> generate_configurations(const SystemConfig& base_config, const ParameterSpace& space);

private:
    SimulationResult execute_configuration(const SystemConfig& config, const Kernel& kernel, const Grid& grid, const KernelResourceRequirements& req, size_t config_idx);
};

} // namespace sim_sm
