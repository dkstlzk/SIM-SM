#include "architecture/sm.hpp"

namespace sim_sm {

SM::SM(size_t sm_id) : sm_id_(sm_id) {}

size_t SM::get_sm_id() const {
    return sm_id_;
}

} // namespace sim_sm
