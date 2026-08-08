#include "runtime/benchmarks.hpp"
#include "architecture/gpu.hpp"
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"
#include "architecture/occupancy.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include "scheduling/greedy_scheduler.hpp"
#include "scheduling/priority_scheduler.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace sim_sm {

sim_sm::Kernel generate_vector_add_kernel(int iterations) {
    std::vector<sim_sm::Instruction> insts;
    for (int i = 0; i < iterations; ++i) {
        int offset = i * 4;
        insts.push_back({sim_sm::Opcode::LOAD, 3, 0, -1, offset});
        insts.push_back({sim_sm::Opcode::LOAD, 4, 1, -1, offset});
        insts.push_back({sim_sm::Opcode::ADD, 5, 3, 4, 0});
        insts.push_back({sim_sm::Opcode::STORE, 2, 5, 2, offset});
    }
    return sim_sm::Kernel("vector_add", insts);
}

sim_sm::Grid build_benchmark_grid(size_t num_threads, size_t block_size, size_t warp_size, bool strided) {
    sim_sm::Grid grid;
    size_t num_blocks = (num_threads + block_size - 1) / block_size;
    if (num_threads == 0) num_blocks = 0;

    for (size_t b = 0; b < num_blocks; ++b) {
        sim_sm::ThreadBlock block(b);
        size_t threads_in_this_block = std::min(block_size, num_threads - b * block_size);
        size_t warps_in_this_block = (threads_in_this_block + warp_size - 1) / warp_size;

        for (size_t w = 0; w < warps_in_this_block; ++w) {
            sim_sm::Warp warp(w);
            size_t threads_in_this_warp = std::min(warp_size, threads_in_this_block - w * warp_size);

            for (size_t l = 0; l < threads_in_this_warp; ++l) {
                size_t local_id = w * warp_size + l;
                size_t global_id = b * block_size + local_id;
                sim_sm::Thread thread(global_id, b, local_id, w, l);

                size_t stride = strided ? 32 : 4;
                thread.registers().write(0, 0 + global_id * stride);
                thread.registers().write(1, 10000 + global_id * stride);
                thread.registers().write(2, 20000 + global_id * stride);

                warp.add_thread(thread);
            }
            warp.set_priority(static_cast<int>(warp.get_warp_id()));
            block.add_warp(warp);
        }
        grid.add_block(block);
    }
    return grid;
}

void run_experiment(const std::string& name, const SystemConfig& config,
                    const std::string& scheduler_type, bool strided,
                    size_t l1_sets = 4, size_t l1_assoc = 4) {

    std::filesystem::create_directories("results");
    std::ofstream out("results/" + name + ".csv");
    out << "Experiment,SMs,TotalCycles,Instructions,IPC,MemInsts,MemTxns,Occupancy,BlocksPerSM" << "\n";

    sim_sm::GPU gpu(config.num_sms, l1_sets, l1_assoc, 16, 8, 32, 1048576);

    for (auto& sm : gpu.get_sms()) {
        if (scheduler_type == "greedy") {
            sm.set_scheduler(std::make_unique<sim_sm::GreedyScheduler>());
        } else if (scheduler_type == "priority") {
            sm.set_scheduler(std::make_unique<sim_sm::PriorityScheduler>());
        } else {
            sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
        }
    }

    size_t num_threads = 2048;
    sim_sm::KernelResourceRequirements req = {10, 0};

    sim_sm::Kernel kernel = generate_vector_add_kernel(100);
    sim_sm::Grid grid = build_benchmark_grid(num_threads, config.block_size, config.warp_size, strided);

    OccupancyResult occ = OccupancyCalculator::compute(config, req);

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;

    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles();
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
    }

    size_t max_cycles = 0;
    for (const auto& sm : gpu.get_sms()) {
        max_cycles = std::max(max_cycles, sm.get_counters().get_cycles());
    }

    double ipc = (max_cycles > 0) ? (double)total_insts / max_cycles : 0.0;

    out << name << "," << config.num_sms << "," << max_cycles << ","
        << total_insts << "," << std::fixed << std::setprecision(2) << ipc << ","
        << total_mem_insts << "," << total_mem_txns << ","
        << occ.occupancy_percentage << "," << occ.resident_blocks << "\n";
    out.close();
}

void run_cache_experiment(const std::string& name, const SystemConfig& config, size_t l1_sets, size_t l1_assoc) {
    std::filesystem::create_directories("results");
    std::ofstream out("results/" + name + ".csv");
    out << "Experiment,SMs,TotalCycles,Instructions,IPC,MemInsts,MemTxns,Occupancy,BlocksPerSM\n";

    // 1 SM to isolate cache behavior without inter-SM interference
    sim_sm::GPU gpu(1, l1_sets, l1_assoc, 16, 8, 32, 1048576);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    size_t num_threads = 64;
    SystemConfig cache_config = config;
    cache_config.num_sms = 1;
    sim_sm::KernelResourceRequirements req = {10, 0};

    std::vector<sim_sm::Instruction> insts;
    for (int iter = 0; iter < 10; ++iter) {
        for (int i = 0; i < 8; ++i) {
            int offset = i * 32;
            insts.push_back({sim_sm::Opcode::LOAD, 3, 0, -1, offset});
        }
    }
    sim_sm::Kernel kernel("cache_stress", insts);
    sim_sm::Grid grid = build_benchmark_grid(num_threads, config.block_size, config.warp_size, false);

    OccupancyResult occ = OccupancyCalculator::compute(cache_config, req);
    gpu.launch_kernel(kernel, grid, cache_config, req);
    gpu.run_to_completion(kernel);

    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;

    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles();
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
    }

    double ipc = (total_cycles > 0) ? (double)total_insts / total_cycles : 0.0;

    out << name << "," << 1 << "," << total_cycles << ","
        << total_insts << "," << std::fixed << std::setprecision(2) << ipc << ","
        << total_mem_insts << "," << total_mem_txns << ","
        << occ.occupancy_percentage << "," << occ.resident_blocks << "\n";
    out.close();
}

sim_sm::Kernel generate_gemm_kernel(int M, int N, int K, int tile_size) {
    std::vector<sim_sm::Instruction> insts;

    // Format: {Opcode, dst, src1, src2, imm}

    // R31 = 4
    insts.push_back({sim_sm::Opcode::MOV, 31, -1, -1, 4});
    // R8 = 0 (K outer counter)
    insts.push_back({sim_sm::Opcode::MOV, 8, -1, -1, 0});
    // R7 = 0 (Accumulator)
    insts.push_back({sim_sm::Opcode::MOV, 7, -1, -1, 0});

    int outer_loop_start = insts.size();

    // A_addr = A_base + (row * K + R8 + tile_col) * 4
    insts.push_back({sim_sm::Opcode::MUL, 20, 0, 13, 0}); // R20 = row * K
    insts.push_back({sim_sm::Opcode::ADD, 21, 20, 8, 0}); // R21 = R20 + R8
    insts.push_back({sim_sm::Opcode::ADD, 22, 21, 3, 0}); // R22 = R21 + tile_col
    insts.push_back({sim_sm::Opcode::MUL, 23, 22, 31, 0});// R23 = R22 * 4
    insts.push_back({sim_sm::Opcode::ADD, 24, 15, 23, 0});// R24 = A_base + R23
    insts.push_back({sim_sm::Opcode::LOAD, 25, 24, -1, 0});// LOAD dst=25, addr=24

    // shared_A_addr = R14 + (tile_row * tile_size + tile_col) * 4
    insts.push_back({sim_sm::Opcode::MUL, 26, 2, 10, 0}); // R26 = tile_row * tile_size
    insts.push_back({sim_sm::Opcode::ADD, 27, 26, 3, 0}); // R27 = R26 + tile_col
    insts.push_back({sim_sm::Opcode::MUL, 28, 27, 31, 0});// R28 = R27 * 4
    insts.push_back({sim_sm::Opcode::ADD, 29, 14, 28, 0});// R29 = R14 + R28

    // STORE [imm], src1 (Wait, let's check InstructionExecutor for STORE)
    insts.push_back({sim_sm::Opcode::STORE, -1, 25, 29, 0}); // We will check STORE

    // B_addr = B_base + ((R8 + tile_row) * N + col) * 4
    insts.push_back({sim_sm::Opcode::ADD, 20, 8, 2, 0});  // R20 = R8 + tile_row
    insts.push_back({sim_sm::Opcode::MUL, 21, 20, 12, 0});// R21 = R20 * N
    insts.push_back({sim_sm::Opcode::ADD, 22, 21, 1, 0}); // R22 = R21 + col
    insts.push_back({sim_sm::Opcode::MUL, 23, 22, 31, 0});// R23 = R22 * 4
    insts.push_back({sim_sm::Opcode::ADD, 24, 16, 23, 0});// R24 = B_base + R23
    insts.push_back({sim_sm::Opcode::LOAD, 25, 24, -1, 0});// LOAD dst=25, addr=24

    // shared_B_addr = R14 + (tile_size * tile_size * 4) + (tile_row * tile_size + tile_col) * 4
    insts.push_back({sim_sm::Opcode::MOV, 30, -1, -1, tile_size * tile_size * 4});
    insts.push_back({sim_sm::Opcode::ADD, 26, 29, 30, 0});// R26 = R29(shared_A_addr) + offset
    insts.push_back({sim_sm::Opcode::STORE, -1, 25, 26, 0});// STORE src1=25, addr=26

    // BARRIER
    insts.push_back({sim_sm::Opcode::BARRIER, 0, 0, 0, 0});

    // Inner loop (R9 = 0 to tile_size)
    insts.push_back({sim_sm::Opcode::MOV, 9, -1, -1, 0});
    int inner_loop_start = insts.size();

    // CMP R9, R10
    insts.push_back({sim_sm::Opcode::CMP, -1, 9, 10, 0});
    // BRANCH inner_loop_exit
    int inner_exit_branch = insts.size();
    insts.push_back({sim_sm::Opcode::BRANCH, -1, -1, -1, 0});

    // shared_A_addr = R14 + (tile_row * tile_size + R9) * 4
    insts.push_back({sim_sm::Opcode::MUL, 20, 2, 10, 0});
    insts.push_back({sim_sm::Opcode::ADD, 21, 20, 9, 0});
    insts.push_back({sim_sm::Opcode::MUL, 22, 21, 31, 0});
    insts.push_back({sim_sm::Opcode::ADD, 23, 14, 22, 0});
    insts.push_back({sim_sm::Opcode::LOAD, 24, 23, -1, 0}); // R24 = A_element

    // shared_B_addr = R14 + R30 + (R9 * tile_size + tile_col) * 4
    insts.push_back({sim_sm::Opcode::MUL, 20, 9, 10, 0});
    insts.push_back({sim_sm::Opcode::ADD, 21, 20, 3, 0});
    insts.push_back({sim_sm::Opcode::MUL, 22, 21, 31, 0});
    insts.push_back({sim_sm::Opcode::ADD, 23, 14, 30, 0});
    insts.push_back({sim_sm::Opcode::ADD, 23, 23, 22, 0});
    insts.push_back({sim_sm::Opcode::LOAD, 25, 23, -1, 0}); // R25 = B_element

    insts.push_back({sim_sm::Opcode::MUL, 26, 24, 25, 0});
    insts.push_back({sim_sm::Opcode::ADD, 7, 7, 26, 0});

    insts.push_back({sim_sm::Opcode::MOV, 27, -1, -1, 1});
    insts.push_back({sim_sm::Opcode::ADD, 9, 9, 27, 0}); // R9++

    // Jump to inner loop start
    insts.push_back({sim_sm::Opcode::CMP, -1, 0, 0, 0}); // R0 == R0
    int jump_inner_start = insts.size();
    insts.push_back({sim_sm::Opcode::BRANCH, -1, -1, -1, inner_loop_start - jump_inner_start});

    int inner_loop_exit = insts.size();
    insts[inner_exit_branch].immediate = inner_loop_exit - inner_exit_branch;

    // BARRIER
    insts.push_back({sim_sm::Opcode::BARRIER, 0, 0, 0, 0});

    insts.push_back({sim_sm::Opcode::ADD, 8, 8, 10, 0}); // R8 += tile_size
    insts.push_back({sim_sm::Opcode::CMP, -1, 8, 13, 0}); // R8 == K ?
    int outer_exit_branch = insts.size();
    insts.push_back({sim_sm::Opcode::BRANCH, -1, -1, -1, 0});

    // Jump to outer loop start
    insts.push_back({sim_sm::Opcode::CMP, -1, 0, 0, 0});
    int jump_outer_start = insts.size();
    insts.push_back({sim_sm::Opcode::BRANCH, -1, -1, -1, outer_loop_start - jump_outer_start});

    int outer_loop_exit = insts.size();
    insts[outer_exit_branch].immediate = outer_loop_exit - outer_exit_branch;

    // Store C
    insts.push_back({sim_sm::Opcode::MUL, 20, 0, 12, 0});
    insts.push_back({sim_sm::Opcode::ADD, 21, 20, 1, 0});
    insts.push_back({sim_sm::Opcode::MUL, 22, 21, 31, 0});
    insts.push_back({sim_sm::Opcode::ADD, 23, 17, 22, 0});
    insts.push_back({sim_sm::Opcode::STORE, -1, 7, 23, 0});

    return sim_sm::Kernel("gemm", insts);
}

sim_sm::Grid build_gemm_grid(int M, int N, int K, int tile_size, int warp_size) {
    sim_sm::Grid grid;
    int blocks_x = N / tile_size;
    int blocks_y = M / tile_size;
    int block_size = tile_size * tile_size;

    size_t A_base = 0;
    size_t B_base = M * K * 4;
    size_t C_base = M * K * 4 + K * N * 4;
    size_t SHARED_MEM_BASE = 0x10000000;

    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            int b = by * blocks_x + bx;
            sim_sm::ThreadBlock block(b);

            int warps_in_block = (block_size + warp_size - 1) / warp_size;
            for (int w = 0; w < warps_in_block; ++w) {
                sim_sm::Warp warp(w);
                int threads_in_warp = std::min(warp_size, block_size - w * warp_size);

                for (int l = 0; l < threads_in_warp; ++l) {
                    int local_id = w * warp_size + l;
                    int tile_row = local_id / tile_size;
                    int tile_col = local_id % tile_size;
                    int global_row = by * tile_size + tile_row;
                    int global_col = bx * tile_size + tile_col;
                    int global_id = global_row * N + global_col;

                    sim_sm::Thread thread(global_id, b, local_id, w, l);

                    thread.registers().write(0, global_row);
                    thread.registers().write(1, global_col);
                    thread.registers().write(2, tile_row);
                    thread.registers().write(3, tile_col);
                    thread.registers().write(4, b);
                    thread.registers().write(10, tile_size);
                    thread.registers().write(11, M);
                    thread.registers().write(12, N);
                    thread.registers().write(13, K);
                    // NOTE: Each GEMM block receives a deterministic shared-memory region
                    // derived from its global block ID. This is sufficient for the current bounded
                    // benchmark but is not a general shared-memory allocation model.
                    thread.registers().write(14, SHARED_MEM_BASE + b * (2 * tile_size * tile_size * 4));
                    thread.registers().write(15, A_base);
                    thread.registers().write(16, B_base);
                    thread.registers().write(17, C_base);

                    warp.add_thread(thread);
                }
                block.add_warp(warp);
            }
            grid.add_block(block);
        }
    }
    return grid;
}

void run_gemm_benchmark(const std::string& name, const SystemConfig& config, int M, int N, int K, int tile_size) {
    std::filesystem::create_directories("results");
    std::ofstream out("results/" + name + ".csv");
    out << "Experiment,SMs,TotalCycles,Instructions,IPC,MemInsts,MemTxns,Occupancy,BlocksPerSM\n";

    sim_sm::GPU gpu(config.num_sms, 16, 4, 16, 8, 32, 1048576);
    for (auto& sm : gpu.get_sms()) {
        sm.set_scheduler(std::make_unique<sim_sm::RoundRobinScheduler>());
    }

    sim_sm::GlobalMemory& gm = gpu.get_global_memory();
    size_t A_base = 0;
    size_t B_base = M * K * 4;
    size_t C_base = M * K * 4 + K * N * 4;

    std::vector<int> cpu_A(M * K);
    std::vector<int> cpu_B(K * N);
    std::vector<int> cpu_C(M * N, 0);

    for (int i = 0; i < M * K; ++i) {
        int val = (i % 5) + 1;
        cpu_A[i] = val;
        gm.store(A_base + i * 4, val);
    }
    for (int i = 0; i < K * N; ++i) {
        int val = (i % 7) + 1;
        cpu_B[i] = val;
        gm.store(B_base + i * 4, val);
    }

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += cpu_A[row * K + k] * cpu_B[k * N + col];
            }
            cpu_C[row * N + col] = sum;
        }
    }

    sim_sm::Kernel kernel = generate_gemm_kernel(M, N, K, tile_size);
    sim_sm::Grid grid = build_gemm_grid(M, N, K, tile_size, config.warp_size);

    sim_sm::KernelResourceRequirements req = {32, (size_t)(tile_size * tile_size * 2 * 4)};
    OccupancyResult occ = OccupancyCalculator::compute(config, req);

    gpu.launch_kernel(kernel, grid, config, req);
    gpu.run_to_completion(kernel);

    bool correct = true;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int gpu_val = gm.load(C_base + (row * N + col) * 4);
            int cpu_val = cpu_C[row * N + col];
            if (gpu_val != cpu_val) {
                correct = false;
                std::cout << "GEMM Error at (" << row << "," << col << "): GPU=" << gpu_val << ", CPU=" << cpu_val << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "GEMM Correctness check passed!" << std::endl;
    } else {
        std::cout << "GEMM Correctness check FAILED!" << std::endl;
    }

    size_t total_cycles = 0;
    size_t total_insts = 0;
    size_t total_mem_insts = 0;
    size_t total_mem_txns = 0;

    for (const auto& sm : gpu.get_sms()) {
        const auto& c = sm.get_counters();
        total_cycles += c.get_cycles();
        total_insts += c.get_instructions_retired();
        total_mem_insts += c.get_memory_instructions();
        total_mem_txns += c.get_memory_transactions();
    }

    size_t max_cycles = 0;
    for (const auto& sm : gpu.get_sms()) {
        max_cycles = std::max(max_cycles, sm.get_counters().get_cycles());
    }

    double ipc = (max_cycles > 0) ? (double)total_insts / max_cycles : 0.0;

    out << name << "," << config.num_sms << "," << max_cycles << ","
        << total_insts << "," << std::fixed << std::setprecision(2) << ipc << ","
        << total_mem_insts << "," << total_mem_txns << ","
        << occ.occupancy_percentage << "," << occ.resident_blocks << "\n";
    out.close();
}

void run_benchmarks(const std::string& config_path, const std::string& b_type) {
    SystemConfig base_config = load_config(config_path);

    bool run_all = (b_type == "all");
    bool run_gemm = run_all || (b_type == "matrix_multiply");
    bool run_basic = run_all || (b_type == "basic");
    bool run_priority = run_all || (b_type == "priority");

    if (!run_all && !run_gemm && !run_basic && !run_priority) {
        throw std::runtime_error("Unknown benchmark: " + b_type);
    }

    if (run_gemm) {
        SystemConfig gemm_config = base_config;
        gemm_config.block_size = 256;
        run_gemm_benchmark("matrix_multiply_32x32_tile16", gemm_config, 32, 32, 32, 16);
    }

    if (run_priority) {
        run_experiment("scheduler_priority", base_config, "priority", false);
    }

    if (run_basic) {
        run_experiment("scheduler_rr", base_config, "rr", false);
        run_experiment("scheduler_greedy", base_config, "greedy", false);
        // By user request, scheduler_priority.csv is generated alongside basic so it's comparable
        if (!run_priority) {
            run_experiment("scheduler_priority", base_config, "priority", false);
        }
        run_cache_experiment("cache_small", base_config, 2, 2);
        run_cache_experiment("cache_large", base_config, 16, 4);
        run_experiment("coalesced", base_config, "rr", false);
        run_experiment("strided", base_config, "rr", true);

        SystemConfig small_block = base_config;
        small_block.block_size = 64;
        run_experiment("occupancy_64", small_block, "rr", false);

        SystemConfig large_block = base_config;
        large_block.block_size = 256;
        run_experiment("occupancy_256", large_block, "rr", false);
    }
}

} // namespace sim_sm
