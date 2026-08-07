#pragma once

#include "architecture/thread.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

enum class WarpState {
    Ready,
    Stalled,
    StalledAtBarrier,
    Completed
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

    void stall(size_t cycles);
    void tick_stall();
    void set_completed();
    void set_stalled_at_barrier();
    void set_ready();

private:
    size_t warp_id_;
    std::vector<Thread> threads_;
    WarpState state_{WarpState::Ready};
    size_t warp_pc_{0};
    size_t stall_cycles_remaining_{0};
};

} // namespace sim_sm
