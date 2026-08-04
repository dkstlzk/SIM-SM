#include <gtest/gtest.h>
#include "memory/cache.hpp"
#include "memory/memory_system.hpp"
#include "memory/shared_memory.hpp"
#include "memory/global_memory.hpp"

namespace sim_sm {

TEST(MemoryTest, CacheHitsAndMisses) {
    // sets = 2, associativity = 2, line_size = 4
    Cache cache(2, 2, 4, "LRU");

    // Address 0 -> Line 0 -> Set 0 (0 % 2)
    EXPECT_FALSE(cache.access(0)); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 1);
    EXPECT_EQ(cache.stats().hits, 0);

    // Address 4 -> Line 1 -> Set 1 (1 % 2)
    EXPECT_FALSE(cache.access(4)); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 2);

    // Address 8 -> Line 2 -> Set 0 (2 % 2)
    EXPECT_FALSE(cache.access(8)); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 3);

    // Address 0 -> Line 0 -> Set 0 (Hit)
    EXPECT_TRUE(cache.access(0));
    EXPECT_EQ(cache.stats().hits, 1);
    
    // Address 4 -> Line 1 -> Set 1 (Hit)
    EXPECT_TRUE(cache.access(4));
    EXPECT_EQ(cache.stats().hits, 2);

    // Address 12 -> Line 3 -> Set 1 (Miss, compulsory)
    EXPECT_FALSE(cache.access(12));
    EXPECT_EQ(cache.stats().misses, 4);

    EXPECT_EQ(cache.stats().evictions, 0); // No capacity evictions yet
}

TEST(MemoryTest, LRUEviction) {
    // sets = 1, associativity = 2, line_size = 4
    Cache cache(1, 2, 4, "LRU");

    EXPECT_FALSE(cache.access(0)); // Line 0 (Tag 0), Miss
    EXPECT_FALSE(cache.access(4)); // Line 1 (Tag 1), Miss

    // Access 0 to make it most recently used
    EXPECT_TRUE(cache.access(0));  // Line 0 (Tag 0), Hit

    // Access 8, maps to Line 2 (Tag 2). Since set has capacity 2, it evicts the LRU.
    // LRU should be Line 1 (Tag 1) because Line 0 was just accessed.
    EXPECT_FALSE(cache.access(8)); // Line 2 (Tag 2), Miss + Eviction
    EXPECT_EQ(cache.stats().evictions, 1);

    // Verify Line 0 is still there (since it was MRU before 8 was accessed)
    EXPECT_TRUE(cache.access(0));  // Line 0 (Tag 0), Hit
    
    // Now LRU is Line 2 (Tag 2). Line 0 is MRU.
    // Verify Line 1 was evicted earlier by accessing it and expecting a miss.
    EXPECT_FALSE(cache.access(4)); // Line 1 (Tag 1), Miss + Eviction (evicts Line 2)
    EXPECT_EQ(cache.stats().evictions, 2);
}

TEST(MemoryTest, MemorySystemAMATAndLatency) {
    SharedMemory shared(65536);
    Cache l1(1, 2, 4); // L1: 1 set, 2 ways, 4 bytes/line
    Cache l2(1, 4, 4); // L2: 1 set, 4 ways, 4 bytes/line
    GlobalMemory global(1000000);
    
    MemoryAccessConfig config;
    config.shared_memory_latency = 1;
    config.l1_latency = 5;
    config.l2_latency = 20;
    config.global_memory_latency = 100;
    
    MemorySystem mem(shared, l1, l2, global, config);

    int val;
    // Shared Memory Access explicitly
    EXPECT_EQ(mem.shared_store(100, 42), config.shared_memory_latency);
    EXPECT_EQ(mem.shared_load(100, val), config.shared_memory_latency);
    EXPECT_EQ(val, 42);

    // Global Memory Access 
    // Access 0 (Miss L1, Miss L2)
    size_t expected_miss_latency = config.l1_latency + config.l2_latency + config.global_memory_latency; // 125
    EXPECT_EQ(mem.store(0, 84), expected_miss_latency);
    
    // Access 0 (Hit L1)
    EXPECT_EQ(mem.load(0, val), config.l1_latency); // 5
    EXPECT_EQ(val, 84);

    // Access 1 (Same cache line as 0 since line_size = 4, hit L1)
    EXPECT_EQ(mem.load(1, val), config.l1_latency); // 5

    // Access 4 (Different line, Miss L1, Miss L2)
    EXPECT_EQ(mem.load(4, val), expected_miss_latency); // 125
    
    // Calculate AMAT
    // Accesses to L1:
    // store(0) -> miss
    // load(0) -> hit
    // load(1) -> hit
    // load(4) -> miss
    // Total L1 accesses = 4 (2 hits, 2 misses). L1 miss rate = 0.5
    
    // Accesses to L2:
    // store(0) -> miss
    // load(4) -> miss
    // Total L2 accesses = 2 (0 hits, 2 misses). L2 miss rate = 1.0

    // AMAT Formula:
    // L2 Miss Penalty = 100 (Global Memory)
    // L1 Miss Penalty = L2 Latency + (L2 Miss Rate * L2 Miss Penalty)
    //                 = 20 + (1.0 * 100) = 120
    // AMAT = L1 Latency + (L1 Miss Rate * L1 Miss Penalty)
    //      = 5 + (0.5 * 120) = 5 + 60 = 65

    EXPECT_DOUBLE_EQ(mem.compute_amat(), 65.0);
    
    // Now let's hit L2 but miss L1
    // Evict 0 from L1 by accessing 8 (associativity is 2)
    // Currently L1 has 0 and 4.
    // Load 8 (Evicts 0 from L1, occupies way 0. Misses L1, Misses L2)
    EXPECT_EQ(mem.load(8, val), expected_miss_latency); // 125

    // Now L1 has 4 and 8. L2 has 0, 4, 8.
    // Access 0: Misses L1, Hits L2
    size_t expected_l2_hit_latency = config.l1_latency + config.l2_latency; // 25
    EXPECT_EQ(mem.load(0, val), expected_l2_hit_latency); // 25

    // Let's recalculate expected AMAT:
    // L1 accesses: 4 + 1(load 8) + 1(load 0) = 6 accesses.
    // L1 misses: 2 + 1(load 8) + 1(load 0) = 4 misses.
    // L1 hits: 2 hits.
    // L1 miss rate = 4 / 6 = 2/3

    // L2 accesses: 2 + 1(load 8) + 1(load 0) = 4 accesses.
    // L2 misses: 2 + 1(load 8) + 0(load 0) = 3 misses.
    // L2 hits: 1 hit.
    // L2 miss rate = 3 / 4 = 0.75

    // L2 Miss Penalty = 100
    // L1 Miss Penalty = 20 + (0.75 * 100) = 95
    // AMAT = 5 + (2/3 * 95) = 5 + 63.333333333333336 = 68.333333333333336

    EXPECT_DOUBLE_EQ(mem.compute_amat(), 5.0 + (2.0 / 3.0) * 95.0);
}

} // namespace sim_sm
