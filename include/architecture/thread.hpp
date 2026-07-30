#pragma once

#include <cstddef>

namespace sim_sm {

class Thread {
public:
    Thread(size_t global_id, size_t block_id, size_t local_id, size_t warp_id, size_t lane_id);

    size_t get_global_id() const;
    size_t get_block_id() const;
    size_t get_local_id() const;
    size_t get_warp_id() const;
    size_t get_lane_id() const;

private:
    size_t global_id_;
    size_t block_id_;
    size_t local_id_;
    size_t warp_id_;
    size_t lane_id_;
};

} // namespace sim_sm
