#include "architecture/warp.hpp"

namespace sim_sm {

Warp::Warp(size_t warp_id) : warp_id_(warp_id) {}

void Warp::add_thread(const Thread& thread) {
    threads_.push_back(thread);
}

std::vector<Thread>& Warp::get_threads() {
    return threads_;
}

const std::vector<Thread>& Warp::get_threads() const {
    return threads_;
}

size_t Warp::get_warp_id() const {
    return warp_id_;
}

WarpState Warp::get_state() const {
    return state_;
}

size_t Warp::get_warp_pc() const {
    return warp_pc_;
}

void Warp::set_warp_pc(size_t pc) {
    warp_pc_ = pc;
}

size_t Warp::get_stall_cycles() const {
    return stall_cycles_remaining_;
}

void Warp::stall(size_t cycles) {
    if (cycles > 0) {
        stall_cycles_remaining_ = cycles;
        state_ = WarpState::Stalled;
    }
}

void Warp::tick_stall() {
    if (state_ == WarpState::Stalled && stall_cycles_remaining_ > 0) {
        stall_cycles_remaining_--;
        if (stall_cycles_remaining_ == 0) {
            state_ = WarpState::Ready;
        }
    }
}

void Warp::set_completed() {
    state_ = WarpState::Completed;
}

void Warp::set_stalled_at_barrier() {
    state_ = WarpState::StalledAtBarrier;
}

void Warp::set_ready() {
    state_ = WarpState::Ready;
}

} // namespace sim_sm
