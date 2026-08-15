#pragma once

#include "architecture/thread.hpp"
#include <vector>
#include <cstddef>
#include <bitset>

namespace sim_sm {

enum class WarpState {
    Ready,
    Stalled,
    StalledAtBarrier,
    Completed
};

struct SIMTStackEntry {
    std::bitset<32> active_mask;
    size_t target_pc;
    size_t reconvergence_pc;
};

class Warp {
public:
    Warp(size_t warp_id);

    void add_thread(const Thread& thread);
    std::vector<Thread>& get_threads();
    const std::vector<Thread>& get_threads() const;
    size_t get_warp_id() const;

    WarpState get_state() const;
    size_t get_warp_pc() const;
    void set_warp_pc(size_t pc);
    size_t get_stall_cycles() const;

    const std::bitset<32>& get_active_mask() const;
    void set_active_mask(const std::bitset<32>& mask);
    
    size_t get_reconvergence_pc() const;
    void set_reconvergence_pc(size_t pc);

    void push_simt_stack(const SIMTStackEntry& entry);
    bool pop_simt_stack(SIMTStackEntry& out_entry);
    bool is_simt_stack_empty() const;

    void stall(size_t cycles);
    void tick_stall(size_t current_cycle);
    void set_completed();
    void set_stalled_at_barrier();
    void set_ready(size_t current_cycle);

    int get_priority() const;
    void set_priority(int priority);

    size_t get_ready_since_cycle() const;
    size_t get_wait_cycles() const;
    void increment_wait_cycles();
    void reset_wait_cycles();
    bool check_and_set_starvation();

private:
    size_t warp_id_;
    std::vector<Thread> threads_;
    WarpState state_{WarpState::Ready};
    size_t warp_pc_{0};
    size_t stall_cycles_remaining_{0};
    int priority_{0};

    size_t ready_since_cycle_{0};
    size_t current_wait_cycles_{0};
    bool starvation_recorded_{false};

    std::bitset<32> active_mask_{0};
    size_t reconvergence_pc_{static_cast<size_t>(-1)};
    std::vector<SIMTStackEntry> simt_stack_;
};

} // namespace sim_sm
