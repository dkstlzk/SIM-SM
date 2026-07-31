#include "architecture/flat_memory.hpp"
#include <stdexcept>

namespace sim_sm {

FlatMemory::FlatMemory(size_t size) : data_(size, 0) {}

int FlatMemory::load(size_t address) const {
    if (address >= data_.size()) {
        throw std::out_of_range("Memory load out of bounds");
    }
    return data_[address];
}

void FlatMemory::store(size_t address, int value) {
    if (address >= data_.size()) {
        throw std::out_of_range("Memory store out of bounds");
    }
    data_[address] = value;
}

} // namespace sim_sm
