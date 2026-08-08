#include "memory/memory_coalescer.hpp"
#include <algorithm>
#include <set>

namespace sim_sm {

std::vector<size_t> MemoryCoalescer::coalesce(const std::vector<size_t>& addresses, size_t line_size) {
    if (line_size == 0) return addresses; // Fallback for zero line size (should not happen)

    std::set<size_t> unique_lines;
    for (size_t addr : addresses) {
        size_t line_base = (addr / line_size) * line_size;
        unique_lines.insert(line_base);
    }

    return std::vector<size_t>(unique_lines.begin(), unique_lines.end());
}

} // namespace sim_sm
