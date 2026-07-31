#pragma once

#include <vector>
#include <cstddef>

namespace sim_sm {

class FlatMemory {
public:
    explicit FlatMemory(size_t size);

    int load(size_t address) const;
    void store(size_t address, int value);

private:
    std::vector<int> data_;
};

} // namespace sim_sm
