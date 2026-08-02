#pragma once

#include <cstddef>
#include <map>

namespace sim_sm {

enum class StallReason {
    None,
    NoReadyWarp,
    SyntheticLatency
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

private:
    size_t cycles_{0};
    size_t instructions_retired_{0};
    size_t stall_cycles_{0};
    std::map<StallReason, size_t> stall_reasons_;
};

} // namespace sim_sm
