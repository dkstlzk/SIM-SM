#pragma once

#include "architecture/thread.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

class Warp {
public:
    Warp(size_t warp_id);

    void add_thread(const Thread& thread);
    const std::vector<Thread>& get_threads() const;
    size_t get_warp_id() const;

private:
    size_t warp_id_;
    std::vector<Thread> threads_;
};

} // namespace sim_sm
