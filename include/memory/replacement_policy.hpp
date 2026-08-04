#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>

namespace sim_sm {

struct CacheLine {
    bool valid = false;
    uint64_t tag = 0;
};

class ReplacementPolicy {
public:
    virtual ~ReplacementPolicy() = default;

    virtual size_t choose_victim(const std::vector<CacheLine>& lines) = 0;
    virtual void on_access(size_t way) = 0;
};

} // namespace sim_sm
