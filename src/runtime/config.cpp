#include "runtime/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace sim_sm {

SystemConfig load_config(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open config file: " + filepath);
    }

    nlohmann::json j;
    ifs >> j;

    SystemConfig config;
    config.num_sms = j.value("num_sms", 4);
    config.warp_size = j.value("warp_size", 32);
    config.block_size = j.value("block_size", 128);

    config.max_threads_per_sm = j.value("max_threads_per_sm", 2048);
    config.max_blocks_per_sm = j.value("max_blocks_per_sm", 32);
    config.max_shared_memory_per_sm = j.value("max_shared_memory_per_sm", 65536);
    config.max_registers_per_sm = j.value("max_registers_per_sm", 65536);
    config.max_warps_per_sm = j.value("max_warps_per_sm",
        config.max_threads_per_sm / config.warp_size);

    // Cache hierarchy
    config.l1_sets = j.value("l1_sets", (size_t)4);
    config.l1_associativity = j.value("l1_associativity", (size_t)4);
    config.l1_line_size = j.value("l1_line_size", (size_t)32);
    config.l1_policy = j.value("l1_policy", std::string("LRU"));

    config.l2_sets = j.value("l2_sets", (size_t)16);
    config.l2_associativity = j.value("l2_associativity", (size_t)8);
    config.l2_line_size = j.value("l2_line_size", (size_t)32);
    config.l2_policy = j.value("l2_policy", std::string("LRU"));

    // Latencies
    config.shared_memory_latency = j.value("shared_memory_latency", (size_t)1);
    config.l1_latency = j.value("l1_latency", (size_t)5);
    config.l2_latency = j.value("l2_latency", (size_t)20);
    config.global_memory_latency = j.value("global_memory_latency", (size_t)100);
    config.writeback_latency = j.value("writeback_latency", (size_t)20);

    // Shared memory banking
    config.shared_memory_banks = j.value("shared_memory_banks", (size_t)32);

    // Global memory size
    config.global_memory_size = j.value("global_memory_size", (size_t)1048576);

    return config;
}

} // namespace sim_sm
