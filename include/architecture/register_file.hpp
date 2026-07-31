#pragma once

#include <vector>
#include <cstddef>

namespace sim_sm {

class RegisterFile {
public:
    explicit RegisterFile(size_t num_registers = 32);

    int read(size_t reg_index) const;
    void write(size_t reg_index, int value);

private:
    std::vector<int> registers_;
};

} // namespace sim_sm
