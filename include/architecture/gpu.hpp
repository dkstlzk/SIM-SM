#pragma once

#include "architecture/sm.hpp"
#include "memory/cache.hpp"
#include "memory/global_memory.hpp"
#include "runtime/config.hpp"
#include "architecture/kernel.hpp"
#include "architecture/grid.hpp"
#include "architecture/occupancy.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

class GPU {
public:
    GPU(size_t num_sms, size_t l1_sets = 4, size_t l1_assoc = 4, size_t l2_sets = 16, size_t l2_assoc = 8, size_t l2_line_size = 32, size_t global_mem_size = 1048576);


    void launch_kernel(const Kernel& kernel, const Grid& grid, const SystemConfig& config, const struct KernelResourceRequirements& req);
    void run_to_completion(const Kernel& kernel);

    std::vector<SM>& get_sms();
    const std::vector<SM>& get_sms() const;

    Cache& get_l2_cache() { return l2_cache_; }
    GlobalMemory& get_global_memory() { return global_memory_; }

private:
    std::vector<SM> sms_;
    Cache l2_cache_;
    GlobalMemory global_memory_;
    std::vector<ThreadBlock> pending_blocks_;
    size_t current_block_idx_ = 0;
    size_t resident_blocks_per_sm_ = 0;
};

} // namespace sim_sm
