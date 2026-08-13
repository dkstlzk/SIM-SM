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
#include "runtime/config.hpp"
#include "architecture/occupancy.hpp"
#include "architecture/thread_block.hpp"
#include <string>

namespace sim_sm {

class WarpScheduler;
class TraceLogger;

struct ResidentBlock {
    size_t block_id;
    size_t num_threads;
    size_t num_warps;
    size_t allocated_registers;
    size_t allocated_shared_memory;
};

class SM {
public:
    SM(size_t sm_id, size_t l1_sets = 4, size_t l1_assoc = 4, size_t l1_line_size = 32, const std::string& cache_policy = "LRU");

    size_t get_sm_id() const;

    void add_warp(const Warp& warp);
    
    bool can_admit(const ThreadBlock& block, const SystemConfig& config, const KernelResourceRequirements& req) const;
    void allocate_block(const ThreadBlock& block, const SystemConfig& config, const KernelResourceRequirements& req);
    void release_completed_blocks();

    void set_scheduler(std::unique_ptr<WarpScheduler> scheduler);
    
    void set_trace_logger(TraceLogger* logger);

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

    size_t get_allocated_blocks() const { return allocated_blocks_; }
    size_t get_allocated_warps() const { return allocated_warps_; }
    size_t get_allocated_threads() const { return allocated_threads_; }
    size_t get_allocated_registers() const { return allocated_registers_; }
    size_t get_allocated_shared_memory() const { return allocated_shared_memory_; }
    const std::vector<ResidentBlock>& get_resident_blocks() const { return resident_blocks_; }

private:
    void handle_barrier_arrival(Warp& warp);

    size_t sm_id_;
    std::vector<Warp> warps_;
    std::unique_ptr<WarpScheduler> scheduler_;
    PerformanceCounter counters_;
    Cache l1_cache_;
    SharedMemory shared_memory_;
    
    size_t allocated_blocks_{0};
    size_t allocated_warps_{0};
    size_t allocated_threads_{0};
    size_t allocated_registers_{0};
    size_t allocated_shared_memory_{0};
    std::vector<ResidentBlock> resident_blocks_;
};

} // namespace sim_sm
