#pragma once
#include <string>
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"

namespace sim_sm {
    void run_benchmarks(const std::string& config_path, const std::string& b_type = "all", bool debug_mode = false, int trace_level = 7, const std::string& trace_file = "");

    sim_sm::Kernel generate_memcpy_kernel(int iterations);
    sim_sm::Grid build_memcpy_grid(size_t num_threads, size_t block_size, size_t warp_size, const std::string& pattern);

    sim_sm::Kernel generate_reduction_kernel(int block_size);
    sim_sm::Grid build_reduction_grid(size_t num_elements, size_t block_size, size_t warp_size);

    sim_sm::Kernel generate_histogram_kernel(int num_bins, int block_size);
    sim_sm::Grid build_histogram_grid(size_t num_elements, size_t block_size, size_t warp_size, size_t num_bins);
}
