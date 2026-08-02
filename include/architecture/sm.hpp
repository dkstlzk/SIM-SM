#pragma once

#include "architecture/warp.hpp"
#include "architecture/kernel.hpp"
#include "architecture/flat_memory.hpp"
#include "scheduling/warp_scheduler.hpp"
#include "runtime/performance_counter.hpp"
#include <cstddef>
#include <vector>
#include <memory>

namespace sim_sm {

class SM {
public:
    SM(size_t sm_id);

    size_t get_sm_id() const;

    void add_warp(const Warp& warp);
    void set_scheduler(std::unique_ptr<WarpScheduler> scheduler);
    
    // Simulate one clock cycle
    void tick(const Kernel& kernel, FlatMemory& memory);

    const PerformanceCounter& get_counters() const;
    std::vector<Warp>& get_warps();
    bool is_completed() const;

private:
    size_t sm_id_;
    std::vector<Warp> warps_;
    std::unique_ptr<WarpScheduler> scheduler_;
    PerformanceCounter counters_;
};

} // namespace sim_sm
