#include "architecture/warp.hpp"

namespace sim_sm {

Warp::Warp(size_t warp_id) : warp_id_(warp_id) {}

void Warp::add_thread(const Thread& thread) {
    size_t lane_id = threads_.size();
    if (lane_id < 32) {
        active_mask_.set(lane_id);
    }
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

const std::bitset<32>& Warp::get_active_mask() const {
    return active_mask_;
}

void Warp::set_active_mask(const std::bitset<32>& mask) {
    active_mask_ = mask;
}

size_t Warp::get_reconvergence_pc() const {
    return reconvergence_pc_;
}

void Warp::set_reconvergence_pc(size_t pc) {
    reconvergence_pc_ = pc;
}

void Warp::push_simt_stack(const SIMTStackEntry& entry) {
    simt_stack_.push_back(entry);
}

bool Warp::pop_simt_stack(SIMTStackEntry& out_entry) {
    if (simt_stack_.empty()) return false;
    out_entry = simt_stack_.back();
    simt_stack_.pop_back();
    return true;
}

bool Warp::is_simt_stack_empty() const {
    return simt_stack_.empty();
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

int Warp::get_priority() const {
    return priority_;
}

void Warp::set_priority(int priority) {
    priority_ = priority;
}

} // namespace sim_sm
