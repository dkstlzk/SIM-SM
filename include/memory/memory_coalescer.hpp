#pragma once

#include <vector>
#include <cstddef>

namespace sim_sm {

class MemoryCoalescer {
public:
    // Groups addresses by cache line identity and returns the unique cache line base addresses.
    // Each returned address represents one memory transaction.
    static std::vector<size_t> coalesce(const std::vector<size_t>& addresses, size_t line_size);
};

} // namespace sim_sm
