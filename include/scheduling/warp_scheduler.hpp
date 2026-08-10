#pragma once

#include "architecture/warp.hpp"
#include <vector>
#include <string>

namespace sim_sm {

class WarpScheduler {
public:
    virtual ~WarpScheduler() = default;

    // Returns a pointer to the selected warp, or nullptr if no warp is ready.
    virtual Warp* select_warp(std::vector<Warp>& warps) = 0;

    virtual std::string name() const = 0;
};

} // namespace sim_sm
