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

TEST(MemoryTest, SharedMemoryRouting) {
    SharedMemory shared(65536);
    Cache l1(1, 2, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);

    MemoryAccessConfig config;
    config.shared_memory_latency = 1;
    config.l1_latency = 5;
    config.l2_latency = 20;
    config.global_memory_latency = 100;

    MemorySystem mem(shared, l1, l2, global, config);

    int val = 0;
    // Store using shared memory base address
    size_t store_lat = mem.store(MemorySystem::SHARED_MEM_BASE + 0x10, 42);
    EXPECT_EQ(store_lat, config.shared_memory_latency);

    size_t load_lat = mem.load(MemorySystem::SHARED_MEM_BASE + 0x10, val);
    EXPECT_EQ(load_lat, config.shared_memory_latency);
    EXPECT_EQ(val, 42);

    // Verify it didn't go to global memory
    EXPECT_EQ(global.load(0x10), 0);

    // Test warp access routing
    std::vector<size_t> addresses = {
        MemorySystem::SHARED_MEM_BASE + 0x20,
        MemorySystem::SHARED_MEM_BASE + 0x24
    };
    std::vector<int> values = { 10, 20 };
    auto res_store = mem.warp_store(addresses, values);
    EXPECT_EQ(res_store.total_latency, config.shared_memory_latency);

    std::vector<int> out_values;
    auto res_load = mem.warp_load(addresses, out_values);
    EXPECT_EQ(res_load.total_latency, config.shared_memory_latency);
    EXPECT_EQ(out_values[0], 10);
    EXPECT_EQ(out_values[1], 20);

    // Verify it didn't hit global memory
    EXPECT_EQ(global.load(0x20), 0);
}

TEST(MemoryTest, MixedAddressSpaceRejection) {
    SharedMemory shared(65536);
    Cache l1(1, 2, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);
    MemoryAccessConfig config;
    MemorySystem mem(shared, l1, l2, global, config);

    std::vector<size_t> mixed_addresses = {
        MemorySystem::SHARED_MEM_BASE + 0x20,
        0x100 // Global memory address
    };
    std::vector<int> values = { 10, 20 };
    std::vector<int> out_values;

    EXPECT_THROW(mem.warp_load(mixed_addresses, out_values), std::runtime_error);
    EXPECT_THROW(mem.warp_store(mixed_addresses, values), std::runtime_error);
}

TEST(MemoryTest, FIFOPolicyEviction) {
    Cache cache(1, 2, 4, "FIFO");

    // Load A (Line 0) and B (Line 1)
    EXPECT_FALSE(cache.access(0)); // A - miss
    EXPECT_FALSE(cache.access(4)); // B - miss

    // FIFO order is now A (oldest), then B (newest)
    
    // Access A again. In LRU this makes A the MRU. In FIFO, A is still the oldest.
    EXPECT_TRUE(cache.access(0)); // A - hit

    // Load C (Line 2). Should evict A, since it's the oldest loaded, despite being recently accessed.
    EXPECT_FALSE(cache.access(8)); // C - miss + eviction
    EXPECT_EQ(cache.stats().evictions, 1);

    // Verify A was evicted (so accessing it misses)
    EXPECT_FALSE(cache.access(0)); // A - miss + eviction (evicts B now)
    EXPECT_EQ(cache.stats().evictions, 2);
}

TEST(MemoryTest, RandomPolicyDeterminism) {
    // Two caches with same seed
    Cache cache1(1, 4, 4, "Random");
    Cache cache2(1, 4, 4, "Random");

    // Fill the 4 ways
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(cache1.access(i * 4));
        EXPECT_FALSE(cache2.access(i * 4));
    }

    // Now cause 10 evictions and verify they are identical
    for (size_t i = 4; i < 14; ++i) {
        EXPECT_FALSE(cache1.access(i * 4));
        EXPECT_FALSE(cache2.access(i * 4));
        
        // Since we can't inspect the cache ways directly, 
        // their deterministic eviction choices imply the valid tags left inside are the same.
        // We will just verify they both have 1 eviction per step.
        EXPECT_EQ(cache1.stats().evictions, i - 3);
        EXPECT_EQ(cache2.stats().evictions, i - 3);
    }
    
    // Verify internal state by querying the same addresses. They should hit/miss identically.
    for (size_t i = 0; i < 14; ++i) {
        EXPECT_EQ(cache1.access(i * 4), cache2.access(i * 4));
    }
}

TEST(MemoryTest, CachePolicyIntegration) {
    // Just verifying the factory creates them without throwing
    EXPECT_NO_THROW(Cache(1, 2, 4, "LRU"));
    EXPECT_NO_THROW(Cache(1, 2, 4, "FIFO"));
    EXPECT_NO_THROW(Cache(1, 2, 4, "Random"));
    EXPECT_THROW(Cache(1, 2, 4, "UnknownPolicy"), std::invalid_argument);
}

} // namespace sim_sm
