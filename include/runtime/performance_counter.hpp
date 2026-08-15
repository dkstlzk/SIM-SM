#pragma once

#include <cstddef>
#include <map>
#include <vector>
#include <string>

namespace sim_sm {

class TraceLogger;
class Warp;
struct Instruction;

enum class StallReason {
    None,
    NoReadyWarp,
    SyntheticLatency,
    WriteConflict
};

class PerformanceCounter {
public:
    void increment_cycles();
    void increment_instructions_retired();
    void add_stall(StallReason reason);

    size_t get_cycles() const;
    size_t get_instructions_retired() const;
    size_t get_stall_cycles() const;
    size_t get_stalls(StallReason reason) const;

    // IPC (simulated warp-instruction IPC)
    double get_ipc() const;

    // Memory tracking
    void increment_memory_instructions();
    void add_memory_transactions(size_t transactions);

    size_t get_memory_instructions() const;
    size_t get_memory_transactions() const;
    double get_transactions_per_memory_instruction() const;

    void add_warp_barrier_stall_cycles(size_t count);
    size_t get_warp_barrier_stall_cycles() const;

    void add_bank_conflict_stalls(size_t count);
    size_t get_bank_conflict_stalls() const;

    void add_dirty_eviction_writebacks(size_t count);
    size_t get_dirty_eviction_writebacks() const;

    // Trace hooks
    void set_trace_logger(TraceLogger* logger);
    void record_scheduler_event(size_t sm_id, size_t cycle, const std::string& scheduler_name, size_t selected_warp_id, const std::vector<Warp>& warps);
    void record_memory_event(size_t sm_id, size_t cycle, size_t warp_id, const Instruction& inst, size_t transactions, const std::string& memory_space);

private:
    size_t cycles_{0};
    size_t instructions_retired_{0};
    size_t stall_cycles_{0};
    std::map<StallReason, size_t> stall_reasons_;
    size_t memory_instructions_{0};
    size_t memory_transactions_{0};
    size_t warp_barrier_stall_cycles_{0};

    size_t bank_conflict_stalls_{0};
    size_t dirty_eviction_writebacks_{0};

    TraceLogger* trace_logger_{nullptr};
};

} // namespace sim_sm
