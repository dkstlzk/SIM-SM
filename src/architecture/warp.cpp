#include "architecture/warp.hpp"

namespace sim_sm {

Warp::Warp(size_t warp_id) : warp_id_(warp_id) {}

void Warp::add_thread(const Thread& thread) {
    threads_.push_back(thread);
}

const std::vector<Thread>& Warp::get_threads() const {
    return threads_;
}

size_t Warp::get_warp_id() const {
    return warp_id_;
}

} // namespace sim_sm
