#include "memory/memory_system.hpp"
#include "memory/memory_coalescer.hpp"
#include <algorithm>
#include <stdexcept>

namespace sim_sm {

size_t MemorySystem::read_latency(size_t address, size_t& dirty_evictions_out) {
    dirty_evictions_out = 0;

    // 1. Check L1 cache
    auto l1_result = l1_cache_.access(address);
    size_t latency = config_.l1_latency;

    if (l1_result.hit) {
        return latency;
    }

    // 2. Check L2 cache on L1 miss
    auto l2_result = l2_cache_.access(address);
    if (l2_result.hit) {
        latency += config_.l2_latency;
    } else {
        latency += config_.l2_latency + config_.global_memory_latency;
    }

    // 3. Dirty L1 eviction adds writeback latency
    if (l1_result.dirty_eviction) {
        latency += config_.writeback_latency;
        dirty_evictions_out = 1;
    }

    return latency;
}

size_t MemorySystem::write_latency(size_t address, size_t& dirty_evictions_out) {
    dirty_evictions_out = 0;

    // 1. Write to L1 (write-allocate + mark dirty)
    auto l1_result = l1_cache_.write(address);

    size_t latency = config_.l1_latency;

    if (!l1_result.hit) {
        // L1 miss: need to fill from L2/Global (demand fill for write-allocate)
        auto l2_result = l2_cache_.access(address);
        if (l2_result.hit) {
            latency += config_.l2_latency;
        } else {
            latency += config_.l2_latency + config_.global_memory_latency;
        }
    }

    // Dirty L1 eviction adds writeback latency (timing only)
    if (l1_result.dirty_eviction) {
        latency += config_.writeback_latency;
        dirty_evictions_out = 1;
    }

    return latency;
}

size_t MemorySystem::load(size_t address, int& out_value) {
    if (address >= SHARED_MEM_BASE) {
        return shared_load(address - SHARED_MEM_BASE, out_value);
    }
    // Global memory access
    size_t dirty_evictions = 0;
    size_t latency = read_latency(address, dirty_evictions);
    out_value = global_memory_.load(address);
    return latency;
}

size_t MemorySystem::store(size_t address, int value) {
    if (address >= SHARED_MEM_BASE) {
        return shared_store(address - SHARED_MEM_BASE, value);
    }
    // Global memory access
    size_t dirty_evictions = 0;
    size_t latency = write_latency(address, dirty_evictions);
    global_memory_.store(address, value);
    return latency;
}

WarpMemoryResult MemorySystem::warp_load(const std::vector<size_t>& addresses, std::vector<int>& out_values) {
    out_values.resize(addresses.size());
    if (addresses.empty()) return {0, 0, 0, 0};

    bool is_shared = (addresses[0] >= SHARED_MEM_BASE);
    for (size_t addr : addresses) {
        if ((addr >= SHARED_MEM_BASE) != is_shared) {
            throw std::runtime_error("Mixed shared/global address spaces within one warp memory instruction");
        }
    }

    if (is_shared) {
        // Compute bank conflicts on local addresses
        std::vector<size_t> local_addresses;
        local_addresses.reserve(addresses.size());
        for (size_t addr : addresses) {
            local_addresses.push_back(addr - SHARED_MEM_BASE);
        }
        size_t bank_conflicts = shared_memory_.compute_bank_conflicts(local_addresses);

        size_t max_latency = 0;
        for (size_t i = 0; i < addresses.size(); ++i) {
            size_t local_addr = addresses[i] - SHARED_MEM_BASE;
            int val;
            size_t lat = shared_load(local_addr, val);
            out_values[i] = val;
            max_latency = std::max(max_latency, lat);
        }
        max_latency += bank_conflicts;
        return {max_latency, addresses.size(), bank_conflicts, 0};
    }

    // Coalesce addresses into cache line transactions
    std::vector<size_t> transactions = MemoryCoalescer::coalesce(addresses, get_l1_line_size());

    size_t max_latency = 1;
    size_t total_dirty_evictions = 0;
    for (size_t line_base : transactions) {
        size_t dirty_evictions = 0;
        size_t latency = read_latency(line_base, dirty_evictions);
        max_latency = std::max(max_latency, latency);
        total_dirty_evictions += dirty_evictions;
    }

    // Actually load the data for each thread
    for (size_t i = 0; i < addresses.size(); ++i) {
        out_values[i] = global_memory_.load(addresses[i]);
    }

    // Total latency: max latency + issue cost for additional transactions (issue cost = 1)
    size_t total_latency = max_latency + (transactions.size() > 0 ? transactions.size() - 1 : 0);

    return {total_latency, transactions.size(), 0, total_dirty_evictions};
}

WarpMemoryResult MemorySystem::warp_store(const std::vector<size_t>& addresses, const std::vector<int>& values) {
    if (addresses.empty()) return {0, 0, 0, 0};

    bool is_shared = (addresses[0] >= SHARED_MEM_BASE);
    for (size_t addr : addresses) {
        if ((addr >= SHARED_MEM_BASE) != is_shared) {
            throw std::runtime_error("Mixed shared/global address spaces within one warp memory instruction");
        }
    }

    if (is_shared) {
        // Compute bank conflicts on local addresses
        std::vector<size_t> local_addresses;
        local_addresses.reserve(addresses.size());
        for (size_t addr : addresses) {
            local_addresses.push_back(addr - SHARED_MEM_BASE);
        }
        size_t bank_conflicts = shared_memory_.compute_bank_conflicts(local_addresses);

        size_t max_latency = 0;
        for (size_t i = 0; i < addresses.size(); ++i) {
            size_t local_addr = addresses[i] - SHARED_MEM_BASE;
            size_t lat = shared_store(local_addr, values[i]);
            max_latency = std::max(max_latency, lat);
        }
        max_latency += bank_conflicts;
        return {max_latency, addresses.size(), bank_conflicts, 0};
    }

    // Coalesce addresses into cache line transactions
    std::vector<size_t> transactions = MemoryCoalescer::coalesce(addresses, get_l1_line_size());

    size_t max_latency = 1;
    size_t total_dirty_evictions = 0;
    for (size_t line_base : transactions) {
        size_t dirty_evictions = 0;
        size_t latency = write_latency(line_base, dirty_evictions);
        max_latency = std::max(max_latency, latency);
        total_dirty_evictions += dirty_evictions;
    }

    // Actually store the data for each thread
    for (size_t i = 0; i < addresses.size(); ++i) {
        global_memory_.store(addresses[i], values[i]);
    }

    // Total latency: max latency + issue cost for additional transactions (issue cost = 1)
    size_t total_latency = max_latency + (transactions.size() > 0 ? transactions.size() - 1 : 0);

    return {total_latency, transactions.size(), 0, total_dirty_evictions};
}

WarpMemoryResult MemorySystem::warp_atomic_add(const std::vector<size_t>& addresses, const std::vector<int>& values) {
    if (addresses.empty()) return {0, 0, 0, 0};

    bool is_shared = (addresses[0] >= SHARED_MEM_BASE);
    for (size_t addr : addresses) {
        if ((addr >= SHARED_MEM_BASE) != is_shared) {
            throw std::runtime_error("Mixed shared/global address spaces within one warp memory instruction");
        }
    }

    if (is_shared) {
        // Compute bank conflicts on local addresses
        std::vector<size_t> local_addresses;
        local_addresses.reserve(addresses.size());
        for (size_t addr : addresses) {
            local_addresses.push_back(addr - SHARED_MEM_BASE);
        }
        size_t bank_conflicts = shared_memory_.compute_bank_conflicts(local_addresses);

        size_t max_latency = 0;
        for (size_t i = 0; i < addresses.size(); ++i) {
            size_t local_addr = addresses[i] - SHARED_MEM_BASE;
            int current_val = shared_memory_.load(local_addr);
            size_t lat = shared_store(local_addr, current_val + values[i]);
            max_latency = std::max(max_latency, lat);
        }
        max_latency += bank_conflicts;
        return {max_latency, addresses.size(), bank_conflicts, 0};
    }

    // Coalesce addresses into cache line transactions
    std::vector<size_t> transactions = MemoryCoalescer::coalesce(addresses, get_l1_line_size());

    size_t max_latency = 1;
    size_t total_dirty_evictions = 0;
    for (size_t line_base : transactions) {
        size_t dirty_evictions = 0;
        size_t latency = write_latency(line_base, dirty_evictions);
        max_latency = std::max(max_latency, latency);
        total_dirty_evictions += dirty_evictions;
    }

    // Actually atomic add the data for each thread sequentially
    for (size_t i = 0; i < addresses.size(); ++i) {
        int current_val = global_memory_.load(addresses[i]);
        global_memory_.store(addresses[i], current_val + values[i]);
    }

    // Total latency: max latency + issue cost for additional transactions (issue cost = 1)
    size_t total_latency = max_latency + (transactions.size() > 0 ? transactions.size() - 1 : 0);

    return {total_latency, transactions.size(), 0, total_dirty_evictions};
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
    double l1_dirty_eviction_rate = (l1_accesses > 0) ? static_cast<double>(l1_stats.dirty_evictions) / l1_accesses : 0.0;

    double miss_penalty_l2 = config_.global_memory_latency;
    double miss_penalty_l1 = config_.l2_latency + (l2_miss_rate * miss_penalty_l2);

    double amat = config_.l1_latency + (l1_miss_rate * miss_penalty_l1) + (l1_dirty_eviction_rate * config_.writeback_latency);

    return amat;
}

size_t MemorySystem::get_l1_line_size() const {
    return l1_cache_.line_size();
}

} // namespace sim_sm
