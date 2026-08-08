#include "runtime/config.hpp"
#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/thread_block.hpp"
#include "architecture/warp.hpp"
#include "architecture/thread.hpp"
#include "runtime/benchmarks.hpp"

#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --config <config.json> --threads <num_threads>\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    // Allow empty grid (0 threads) per requirements, but default to uninitialized (-1) to check if passed
    long long num_threads_arg = -1;

    bool run_benchmarks = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads_arg = std::stoll(argv[++i]);
        } else if (arg == "--benchmark") {
            std::string b_type = "all";
            if (i + 1 < argc) {
                b_type = argv[++i];
            }
            if (config_path.empty()) {
                std::cerr << "Error: --config must precede --benchmark\n";
                return EXIT_FAILURE;
            }
            try {
                sim_sm::run_benchmarks(config_path, b_type);
                std::cout << "Benchmarks completed successfully. Results written to /results/\n";
                return EXIT_SUCCESS;
            } catch (const std::exception& e) {
                std::cerr << "Benchmark error: " << e.what() << "\n";
                return EXIT_FAILURE;
            }
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (num_threads_arg < 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    size_t num_threads = static_cast<size_t>(num_threads_arg);

    try {
        sim_sm::SystemConfig config = sim_sm::load_config(config_path);

        sim_sm::GPU gpu(config.num_sms);
        sim_sm::Grid grid;

        size_t block_size = config.block_size;
        size_t warp_size = config.warp_size;

        size_t num_blocks = (num_threads + block_size - 1) / block_size;
        if (num_threads == 0) {
            num_blocks = 0; // Empty grid edge case
        }

        size_t total_warps = 0;

        for (size_t b = 0; b < num_blocks; ++b) {
            sim_sm::ThreadBlock block(b);

            size_t threads_in_this_block = std::min(block_size, num_threads - b * block_size);
            size_t warps_in_this_block = (threads_in_this_block + warp_size - 1) / warp_size;

            for (size_t w = 0; w < warps_in_this_block; ++w) {
                size_t local_warp_id = w;
                sim_sm::Warp warp(local_warp_id);

                size_t threads_in_this_warp = std::min(warp_size, threads_in_this_block - w * warp_size);

                for (size_t l = 0; l < threads_in_this_warp; ++l) {
                    size_t local_id = w * warp_size + l;
                    size_t global_id = b * block_size + local_id;
                    size_t lane_id = l;
                    sim_sm::Thread thread(global_id, b, local_id, local_warp_id, lane_id);
                    warp.add_thread(thread);
                }

                block.add_warp(warp);
                total_warps++;
            }
            grid.add_block(block);
        }

        std::cout << "Grid: " << num_threads << " threads -> "
                  << total_warps << " warps -> "
                  << grid.get_blocks().size() << " blocks -> scheduled across "
                  << gpu.get_sms().size() << " SMs\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
