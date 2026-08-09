#include "memory/fifo_policy.hpp"
#include <stdexcept>

namespace sim_sm {

FIFOPolicy::FIFOPolicy(size_t associativity) 
    : associativity_(associativity), in_queue_(associativity, false) {}

size_t FIFOPolicy::choose_victim(const std::vector<CacheLine>& lines) {
    // Check for an invalid line first
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].valid) {
            return i;
        }
    }

    if (fifo_queue_.empty()) {
        throw std::runtime_error("FIFO queue is empty but cache is full");
    }

    // Evict the oldest
    size_t victim = fifo_queue_.front();
    fifo_queue_.pop();
    in_queue_[victim] = false;
    
    return victim;
}

void FIFOPolicy::on_access(size_t way) {
    // A cache hit should not refresh age.
    // If it's a new allocation (or was just evicted and reassigned), 
    // it won't be in the queue, so we push it.
    if (!in_queue_[way]) {
        fifo_queue_.push(way);
        in_queue_[way] = true;
    }
}

} // namespace sim_sm
