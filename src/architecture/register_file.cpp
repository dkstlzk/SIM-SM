#include "architecture/register_file.hpp"
#include <stdexcept>

namespace sim_sm {

RegisterFile::RegisterFile(size_t num_registers) : registers_(num_registers, 0) {}

int RegisterFile::read(size_t reg_index) const {
    if (reg_index >= registers_.size()) {
        throw std::out_of_range("Register read out of bounds");
    }
    return registers_[reg_index];
}

void RegisterFile::write(size_t reg_index, int value) {
    if (reg_index >= registers_.size()) {
        throw std::out_of_range("Register write out of bounds");
    }
    registers_[reg_index] = value;
}

} // namespace sim_sm
