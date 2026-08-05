#include "runtime/performance_counter.hpp"

namespace sim_sm {

void PerformanceCounter::increment_cycles() {
    cycles_++;
}

void PerformanceCounter::increment_instructions_retired() {
    instructions_retired_++;
}

void PerformanceCounter::add_stall(StallReason reason) {
    stall_cycles_++;
    stall_reasons_[reason]++;
}

size_t PerformanceCounter::get_cycles() const {
    return cycles_;
}

size_t PerformanceCounter::get_instructions_retired() const {
    return instructions_retired_;
}

size_t PerformanceCounter::get_stall_cycles() const {
    return stall_cycles_;
}

size_t PerformanceCounter::get_stalls(StallReason reason) const {
    auto it = stall_reasons_.find(reason);
    if (it != stall_reasons_.end()) {
        return it->second;
    }
    return 0;
}

double PerformanceCounter::get_ipc() const {
    if (cycles_ == 0) return 0.0;
    return static_cast<double>(instructions_retired_) / cycles_;
}

void PerformanceCounter::increment_memory_instructions() {
    memory_instructions_++;
}

void PerformanceCounter::add_memory_transactions(size_t transactions) {
    memory_transactions_ += transactions;
}

size_t PerformanceCounter::get_memory_instructions() const {
    return memory_instructions_;
}

size_t PerformanceCounter::get_memory_transactions() const {
    return memory_transactions_;
}

double PerformanceCounter::get_transactions_per_memory_instruction() const {
    if (memory_instructions_ == 0) return 0.0;
    return static_cast<double>(memory_transactions_) / memory_instructions_;
}

} // namespace sim_sm
