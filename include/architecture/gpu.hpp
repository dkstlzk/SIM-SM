#pragma once

#include "architecture/sm.hpp"
#include "memory/cache.hpp"
#include "memory/global_memory.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

class GPU {
public:
    GPU(size_t num_sms, size_t l2_sets = 16, size_t l2_assoc = 8, size_t l2_line_size = 32, size_t global_mem_size = 1048576);

    std::vector<SM>& get_sms();
    const std::vector<SM>& get_sms() const;

    Cache& get_l2_cache() { return l2_cache_; }
    GlobalMemory& get_global_memory() { return global_memory_; }

private:
    std::vector<SM> sms_;
    Cache l2_cache_;
    GlobalMemory global_memory_;
};

} // namespace sim_sm
