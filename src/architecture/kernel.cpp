#include "architecture/kernel.hpp"
#include <utility>

namespace sim_sm {

Kernel::Kernel(std::string name) : name_(std::move(name)) {}

const std::string& Kernel::name() const {
    return name_;
}

} // namespace sim_sm
