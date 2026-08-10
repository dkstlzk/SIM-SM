#pragma once

#include "replacement_policy.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>

namespace sim_sm {

struct CacheStats {
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

struct CacheEvent {
    size_t cycle;
    std::string cache_name;
    size_t set;
    size_t way;
    bool hit;
};

class Cache {
public:
    using EventCallback = std::function<void(const CacheEvent&)>;

    // Factory method to allow dependency injection of ReplacementPolicy
    Cache(size_t num_sets, size_t associativity, size_t line_size, const std::string& policy_name = "LRU");

    void set_name(const std::string& name) { name_ = name; }
    void set_event_callback(EventCallback cb) { event_cb_ = cb; }

    // Returns true on hit, false on miss
    bool access(size_t address, size_t cycle = 0);

    const CacheStats& stats() const { return stats_; }
    size_t line_size() const { return line_size_; }

private:
    size_t num_sets_;
    size_t associativity_;
    size_t line_size_;
    std::string name_{"Cache"};
    
    EventCallback event_cb_;

    CacheStats stats_;

    std::vector<std::vector<CacheLine>> sets_;
    std::vector<std::unique_ptr<ReplacementPolicy>> policies_;
};

} // namespace sim_sm
