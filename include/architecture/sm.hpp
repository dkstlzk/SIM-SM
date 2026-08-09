#pragma once

#include "architecture/warp.hpp"
#include "architecture/kernel.hpp"
#include "architecture/flat_memory.hpp"
#include "scheduling/warp_scheduler.hpp"
#include "runtime/performance_counter.hpp"
#include <cstddef>
#include <vector>
#include <memory>

#include "memory/cache.hpp"
#include "memory/memory_system.hpp"
#include "memory/shared_memory.hpp"

namespace sim_sm {

class SM {
public:
    SM(size_t sm_id, size_t l1_sets = 4, size_t l1_assoc = 4, size_t l1_line_size = 32, const std::string& cache_policy = "LRU");

    size_t get_sm_id() const;

    void add_warp(const Warp& warp);
    void set_scheduler(std::unique_ptr<WarpScheduler> scheduler);
    
    // Simulate one clock cycle
    void tick(const Kernel& kernel, MemorySystem& memory);

    const PerformanceCounter& get_counters() const;
    std::vector<Warp>& get_warps();
    bool is_completed() const;
    void clear_warps();
    
    
    Cache& get_l1_cache() { return l1_cache_; }
    const Cache& get_l1_cache() const { return l1_cache_; }

    SharedMemory& get_shared_memory() { return shared_memory_; }
    const SharedMemory& get_shared_memory() const { return shared_memory_; }

private:
    void handle_barrier_arrival(Warp& warp);

    size_t sm_id_;
    std::vector<Warp> warps_;
    std::unique_ptr<WarpScheduler> scheduler_;
    PerformanceCounter counters_;
    Cache l1_cache_;
    SharedMemory shared_memory_;
};

} // namespace sim_sm
