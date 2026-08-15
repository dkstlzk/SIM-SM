#include "runtime/performance_counter.hpp"
#include "runtime/trace_logger.hpp"

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

void PerformanceCounter::add_warp_barrier_stall_cycles(size_t count) {
    warp_barrier_stall_cycles_ += count;
}

size_t PerformanceCounter::get_warp_barrier_stall_cycles() const {
    return warp_barrier_stall_cycles_;
}

void PerformanceCounter::record_warp_wait(size_t wait_cycles) {
    total_warp_wait_cycles_ += wait_cycles;
    if (wait_cycles > max_warp_wait_cycles_) {
        max_warp_wait_cycles_ = wait_cycles;
    }
    if (wait_cycles > 0) {
        warp_wait_events_++;
    }
}

void PerformanceCounter::record_starvation_event() {
    starvation_events_++;
}

void PerformanceCounter::initialize_warp_issue_count(size_t warp_id) {
    if (warp_issue_counts_.find(warp_id) == warp_issue_counts_.end()) {
        warp_issue_counts_[warp_id] = 0;
    }
}

void PerformanceCounter::record_warp_issue(size_t warp_id) {
    warp_issue_counts_[warp_id]++;
}

double PerformanceCounter::get_jains_fairness_index() const {
    if (warp_issue_counts_.empty()) return 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& pair : warp_issue_counts_) {
        sum += pair.second;
        sum_sq += pair.second * pair.second;
    }
    if (sum_sq == 0.0) return 0.0; // Avoid divide by zero if no issues happened at all
    double n = warp_issue_counts_.size();
    return (sum * sum) / (n * sum_sq);
}

double PerformanceCounter::get_average_warp_wait_cycles() const {
    if (warp_wait_events_ == 0) return 0.0;
    return static_cast<double>(total_warp_wait_cycles_) / warp_wait_events_;
}

size_t PerformanceCounter::get_max_warp_wait_cycles() const { return max_warp_wait_cycles_; }
size_t PerformanceCounter::get_starvation_events() const { return starvation_events_; }
size_t PerformanceCounter::get_warp_wait_events() const { return warp_wait_events_; }
size_t PerformanceCounter::get_total_warp_wait_cycles() const { return total_warp_wait_cycles_; }

void PerformanceCounter::set_trace_logger(TraceLogger* logger) {
    trace_logger_ = logger;
}

void PerformanceCounter::record_scheduler_event(size_t sm_id, size_t cycle, const std::string& scheduler_name, size_t selected_warp_id, const std::vector<Warp>& warps) {
    if (trace_logger_) {
        trace_logger_->log_scheduler_event(sm_id, cycle, scheduler_name, selected_warp_id, warps);
    }
}

void PerformanceCounter::record_memory_event(size_t sm_id, size_t cycle, size_t warp_id, const Instruction& inst, size_t transactions, const std::string& memory_space) {
    if (trace_logger_) {
        trace_logger_->log_memory_event(sm_id, cycle, warp_id, inst, transactions, memory_space);
    }
}

void PerformanceCounter::add_bank_conflict_stalls(size_t count) {
    bank_conflict_stalls_ += count;
}

size_t PerformanceCounter::get_bank_conflict_stalls() const {
    return bank_conflict_stalls_;
}

void PerformanceCounter::add_dirty_eviction_writebacks(size_t count) {
    dirty_eviction_writebacks_ += count;
}

size_t PerformanceCounter::get_dirty_eviction_writebacks() const {
    return dirty_eviction_writebacks_;
}

} // namespace sim_sm
