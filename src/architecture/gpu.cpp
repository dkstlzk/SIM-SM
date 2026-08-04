#include "architecture/gpu.hpp"

namespace sim_sm {

GPU::GPU(size_t num_sms, size_t l2_sets, size_t l2_assoc, size_t l2_line_size, size_t global_mem_size)
    : l2_cache_(l2_sets, l2_assoc, l2_line_size, "LRU"), global_memory_(global_mem_size) {
    for (size_t i = 0; i < num_sms; ++i) {
        sms_.emplace_back(i);
    }
}

std::vector<SM>& GPU::get_sms() {
    return sms_;
}

const std::vector<SM>& GPU::get_sms() const {
    return sms_;
}

} // namespace sim_sm
