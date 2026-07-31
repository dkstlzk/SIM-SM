#include "architecture/kernel.hpp"
#include <utility>

namespace sim_sm {

Kernel::Kernel(std::string name, std::vector<Instruction> instructions) 
    : name_(std::move(name))
    , instructions_(std::move(instructions)) {}

const std::string& Kernel::name() const {
    return name_;
}

const std::vector<Instruction>& Kernel::instructions() const {
    return instructions_;
}

} // namespace sim_sm
