#include "memory/memory_system.hpp"

namespace sim_sm {

size_t MemorySystem::access_latency(size_t address) {
    // 1. Check L1 cache
    if (l1_cache_.access(address)) {
        return config_.l1_latency;
    }

    // 2. Check L2 cache on L1 miss
    if (l2_cache_.access(address)) {
        return config_.l1_latency + config_.l2_latency;
    }

    // 3. Fallback to Global Memory on L2 miss
    return config_.l1_latency + config_.l2_latency + config_.global_memory_latency;
}

size_t MemorySystem::load(size_t address, int& out_value) {
    // Global memory access
    size_t latency = access_latency(address);
    out_value = global_memory_.load(address);
    return latency;
}

size_t MemorySystem::store(size_t address, int value) {
    // Global memory access
    size_t latency = access_latency(address);
    global_memory_.store(address, value);
    return latency;
}

size_t MemorySystem::shared_load(size_t address, int& out_value) {
    out_value = shared_memory_.load(address);
    return config_.shared_memory_latency;
}

size_t MemorySystem::shared_store(size_t address, int value) {
    shared_memory_.store(address, value);
    return config_.shared_memory_latency;
}

double MemorySystem::compute_amat() const {
    const auto& l1_stats = l1_cache_.stats();
    const auto& l2_stats = l2_cache_.stats();

    size_t l1_accesses = l1_stats.hits + l1_stats.misses;
    size_t l2_accesses = l2_stats.hits + l2_stats.misses;

    double l1_miss_rate = (l1_accesses > 0) ? static_cast<double>(l1_stats.misses) / l1_accesses : 0.0;
    double l2_miss_rate = (l2_accesses > 0) ? static_cast<double>(l2_stats.misses) / l2_accesses : 0.0;

    double miss_penalty_l2 = config_.global_memory_latency;
    double miss_penalty_l1 = config_.l2_latency + (l2_miss_rate * miss_penalty_l2);
    
    double amat = config_.l1_latency + (l1_miss_rate * miss_penalty_l1);
    
    return amat;
}

} // namespace sim_sm
