#pragma once

#include <string>
#include <cstddef>

namespace sim_sm {

struct SystemConfig {
    size_t num_sms;
    size_t warp_size;
    size_t block_size;
};

SystemConfig load_config(const std::string& filepath);

} // namespace sim_sm
