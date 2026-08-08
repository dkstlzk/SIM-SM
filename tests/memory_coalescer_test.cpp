#include <gtest/gtest.h>
#include "memory/memory_coalescer.hpp"
#include <vector>

using namespace sim_sm;

TEST(MemoryCoalescerTest, CoalescedPattern) {
    // 32 threads accessing contiguous words (4 bytes each)
    std::vector<size_t> addresses;
    for (int i = 0; i < 32; ++i) {
        addresses.push_back(i * 4); // 0, 4, 8, ... 124
    }

    // Line size 128 bytes -> exactly 1 cache line required (Base 0)
    auto trans_128 = MemoryCoalescer::coalesce(addresses, 128);
    EXPECT_EQ(trans_128.size(), 1);
    EXPECT_EQ(trans_128[0], 0);

    // Line size 32 bytes -> 4 cache lines required (Bases 0, 32, 64, 96)
    auto trans_32 = MemoryCoalescer::coalesce(addresses, 32);
    EXPECT_EQ(trans_32.size(), 4);
}

TEST(MemoryCoalescerTest, StridedPattern) {
    // 32 threads accessing with a stride of 32 bytes
    std::vector<size_t> addresses;
    for (int i = 0; i < 32; ++i) {
        addresses.push_back(i * 32); // 0, 32, 64, ... 992
    }

    // Line size 32 bytes -> 32 cache lines required
    auto trans_32 = MemoryCoalescer::coalesce(addresses, 32);
    EXPECT_EQ(trans_32.size(), 32);

    // Line size 128 bytes -> 32 accesses * 32 bytes/access = 1024 bytes spanning 8 cache lines
    auto trans_128 = MemoryCoalescer::coalesce(addresses, 128);
    EXPECT_EQ(trans_128.size(), 8);
}

TEST(MemoryCoalescerTest, ScatteredPattern) {
    // 32 threads accessing widely scattered addresses
    std::vector<size_t> addresses;
    for (int i = 0; i < 32; ++i) {
        addresses.push_back(i * 1024); // 0, 1024, 2048, ...
    }

    // Line size 32 bytes -> 32 cache lines required
    auto trans_32 = MemoryCoalescer::coalesce(addresses, 32);
    EXPECT_EQ(trans_32.size(), 32);

    // Line size 128 bytes -> still 32 cache lines required
    auto trans_128 = MemoryCoalescer::coalesce(addresses, 128);
    EXPECT_EQ(trans_128.size(), 32);
}

TEST(MemoryCoalescerTest, BoundaryCrossing) {
    std::vector<size_t> addresses = {28, 32, 36}; // Three words

    // Line size 32 -> should cross boundary between 0 and 32
    auto trans_32 = MemoryCoalescer::coalesce(addresses, 32);
    EXPECT_EQ(trans_32.size(), 2);

    bool has_0 = false, has_32 = false;
    for (auto line : trans_32) {
        if (line == 0) has_0 = true;
        if (line == 32) has_32 = true;
    }
    EXPECT_TRUE(has_0);
    EXPECT_TRUE(has_32);
}
