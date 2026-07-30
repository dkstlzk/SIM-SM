#pragma once

#include <cstddef>

namespace sim_sm {

class SM {
public:
    SM(size_t sm_id);

    size_t get_sm_id() const;

private:
    size_t sm_id_;
};

} // namespace sim_sm
