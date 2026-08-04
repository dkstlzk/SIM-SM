#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace sim_sm {

class GlobalMemory {
public:
    explicit GlobalMemory(size_t size) : data_(size, 0) {}

    int load(size_t address) const {
        if (address >= data_.size()) {
            throw std::out_of_range("GlobalMemory address out of bounds");
        }
        return data_[address];
    }

    void store(size_t address, int value) {
        if (address >= data_.size()) {
            throw std::out_of_range("GlobalMemory address out of bounds");
        }
        data_[address] = value;
    }

private:
    std::vector<int> data_;
};

} // namespace sim_sm
