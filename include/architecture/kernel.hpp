#pragma once

#include "instruction/instruction.hpp"
#include <string>
#include <vector>

namespace sim_sm {

class Kernel {
public:
    Kernel(std::string name, std::vector<Instruction> instructions = {});

    const std::string& name() const;
    const std::vector<Instruction>& instructions() const;

private:
    std::string name_;
    std::vector<Instruction> instructions_;
};

} // namespace sim_sm
