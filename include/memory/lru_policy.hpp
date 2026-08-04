#pragma once

#include "replacement_policy.hpp"
#include <list>
#include <unordered_map>

namespace sim_sm {

class LRUPolicy : public ReplacementPolicy {
public:
    explicit LRUPolicy(size_t associativity);

    size_t choose_victim(const std::vector<CacheLine>& lines) override;
    void on_access(size_t way) override;

private:
    size_t associativity_;
    // Front is most recently used, back is least recently used
    std::list<size_t> lru_list_;
};

} // namespace sim_sm
