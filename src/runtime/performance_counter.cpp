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

} // namespace sim_sm
