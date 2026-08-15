#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace sim_sm {

class SharedMemory {
public:
    explicit SharedMemory(size_t size, size_t num_banks = 32)
        : data_(size, 0), num_banks_(num_banks) {}

    int load(size_t address) const {
        if (address >= data_.size()) {
            throw std::out_of_range("SharedMemory address out of bounds");
        }
        return data_[address];
    }

    void store(size_t address, int value) {
        if (address >= data_.size()) {
            throw std::out_of_range("SharedMemory address out of bounds");
        }
        data_[address] = value;
    }

    // Compute bank conflict penalty for a set of warp addresses.
    // Returns the number of additional serialization cycles.
    // Broadcasts (same word address) are free.
    size_t compute_bank_conflicts(const std::vector<size_t>& addresses) const {
        if (addresses.empty() || num_banks_ == 0) return 0;

        // Map: bank -> set of unique word addresses
        std::unordered_map<size_t, std::unordered_set<size_t>> bank_words;

        for (size_t addr : addresses) {
            size_t word_addr = addr / 4;
            size_t bank = word_addr % num_banks_;
            bank_words[bank].insert(word_addr);
        }

        size_t max_unique = 0;
        for (const auto& [bank, words] : bank_words) {
            max_unique = std::max(max_unique, words.size());
        }

        return (max_unique > 0) ? max_unique - 1 : 0;
    }

    size_t num_banks() const { return num_banks_; }

private:
    std::vector<int> data_;
    size_t num_banks_;
};

} // namespace sim_sm
