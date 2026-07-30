#pragma once

#include <string>

namespace sim_sm {

class Kernel {
public:
    explicit Kernel(std::string name);

    const std::string& name() const;

private:
    std::string name_;
};

} // namespace sim_sm
