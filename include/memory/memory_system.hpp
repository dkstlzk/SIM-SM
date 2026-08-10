#pragma once

#include "memory/cache.hpp"
#include "memory/global_memory.hpp"
#include "memory/shared_memory.hpp"
#include <cstddef>

namespace sim_sm {

class Cache;

struct MemoryAccessConfig {
    size_t shared_memory_latency = 1;
    size_t l1_latency = 5;
    size_t l2_latency = 20;
    size_t global_memory_latency = 100;
};

struct WarpMemoryResult {
    size_t total_latency;
    size_t num_transactions;
};

class MemorySystem {
public:
    static constexpr size_t SHARED_MEM_BASE = 0x10000000;

    MemorySystem(SharedMemory& shared_memory, Cache& l1_cache, Cache& l2_cache, GlobalMemory& global_memory, const MemoryAccessConfig& config = {})
        : shared_memory_(shared_memory), l1_cache_(l1_cache), l2_cache_(l2_cache), global_memory_(global_memory), config_(config) {}

    // Simulates a global memory load. Returns the simulated latency in cycles.
    size_t load(size_t address, int& out_value);

    // Simulates a global memory store. Returns the simulated latency in cycles.
    size_t store(size_t address, int value);

    // Warp-level global memory accesses (coalesced)
    WarpMemoryResult warp_load(const std::vector<size_t>& addresses, std::vector<int>& out_values);
    WarpMemoryResult warp_store(const std::vector<size_t>& addresses, const std::vector<int>& values);

    // Explicitly access shared memory (not part of the global load/store ISA for Week 1)
    size_t shared_load(size_t address, int& out_value);
    size_t shared_store(size_t address, int value);

    // AMAT Calculation
    double compute_amat() const;

    size_t get_l1_line_size() const;

private:
    SharedMemory& shared_memory_;
    Cache& l1_cache_;
    Cache& l2_cache_;
    GlobalMemory& global_memory_;
    MemoryAccessConfig config_;

    size_t access_latency(size_t address);
};

} // namespace sim_sm
