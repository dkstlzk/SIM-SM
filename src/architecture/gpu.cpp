#include "architecture/gpu.hpp"

namespace sim_sm {

GPU::GPU(size_t num_sms) {
    for (size_t i = 0; i < num_sms; ++i) {
        sms_.emplace_back(i);
    }
}

const std::vector<SM>& GPU::get_sms() const {
    return sms_;
}

} // namespace sim_sm
