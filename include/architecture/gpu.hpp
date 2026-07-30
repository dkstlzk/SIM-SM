#pragma once

#include "architecture/sm.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

class GPU {
public:
    GPU(size_t num_sms);

    const std::vector<SM>& get_sms() const;

private:
    std::vector<SM> sms_;
};

} // namespace sim_sm
