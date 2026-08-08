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

    return config;
}

} // namespace sim_sm
