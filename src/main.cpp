#include "runtime/config.hpp"
#include "architecture/gpu.hpp"
#include "architecture/grid.hpp"
#include "architecture/thread_block.hpp"
#include "architecture/warp.hpp"
#include "architecture/thread.hpp"
#include "runtime/benchmarks.hpp"
#include "runtime/trace_logger.hpp"
#include <memory>

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

    bool debug_mode = false;
    int trace_level_val = 7; // Default to All if --debug is passed
    std::string trace_file = "";

    bool run_benchmarks = false;
    std::string b_type = "all";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads_arg = std::stoll(argv[++i]);
        } else if (arg == "--debug") {
            debug_mode = true;
        } else if (arg == "--trace-level" && i + 1 < argc) {
            trace_level_val = std::stoi(argv[++i]);
        } else if (arg == "--trace-file" && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmarks = true;
            if (i + 1 < argc) {
                // Peek at the next argument. If it doesn't start with "--", it's the benchmark type
                std::string next_arg = argv[i+1];
                if (next_arg.substr(0, 2) != "--") {
                    b_type = next_arg;
                    i++;
                }
            }
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (config_path.empty()) {
        std::cerr << "Error: --config is required\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (run_benchmarks) {
        try {
            sim_sm::run_benchmarks(config_path, b_type, debug_mode, trace_level_val, trace_file);
            std::cout << "Benchmarks completed successfully. Results written to /results/\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& e) {
            std::cerr << "Benchmark error: " << e.what() << "\n";
            return EXIT_FAILURE;
        }
    }
    if (num_threads_arg < 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    size_t num_threads = static_cast<size_t>(num_threads_arg);

    try {
        sim_sm::SystemConfig config = sim_sm::load_config(config_path);

        sim_sm::GPU gpu(config.num_sms);
        std::unique_ptr<sim_sm::TraceLogger> logger;
        if (debug_mode) {
            logger = std::make_unique<sim_sm::TraceLogger>(static_cast<sim_sm::TraceLevel>(trace_level_val), trace_file);
            gpu.set_trace_logger(logger.get());
        }

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
