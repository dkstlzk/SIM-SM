#include <gtest/gtest.h>
#include "memory/cache.hpp"
#include "memory/memory_system.hpp"
#include "memory/shared_memory.hpp"
#include "memory/global_memory.hpp"
#include "runtime/config.hpp"
#include "architecture/gpu.hpp"

namespace sim_sm {

TEST(MemoryTest, CacheHitsAndMisses) {
    // sets = 2, associativity = 2, line_size = 4
    Cache cache(2, 2, 4, "LRU");

    // Address 0 -> Line 0 -> Set 0 (0 % 2)
    EXPECT_FALSE(cache.access(0).hit); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 1);
    EXPECT_EQ(cache.stats().hits, 0);

    // Address 4 -> Line 1 -> Set 1 (1 % 2)
    EXPECT_FALSE(cache.access(4).hit); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 2);

    // Address 8 -> Line 2 -> Set 0 (2 % 2)
    EXPECT_FALSE(cache.access(8).hit); // Miss, compulsory
    EXPECT_EQ(cache.stats().misses, 3);

    // Address 0 -> Line 0 -> Set 0 (Hit)
    EXPECT_TRUE(cache.access(0).hit);
    EXPECT_EQ(cache.stats().hits, 1);

    // Address 4 -> Line 1 -> Set 1 (Hit)
    EXPECT_TRUE(cache.access(4).hit);
    EXPECT_EQ(cache.stats().hits, 2);

    // Address 12 -> Line 3 -> Set 1 (Miss, compulsory)
    EXPECT_FALSE(cache.access(12).hit);
    EXPECT_EQ(cache.stats().misses, 4);

    EXPECT_EQ(cache.stats().evictions, 0); // No capacity evictions yet
}

TEST(MemoryTest, LRUEviction) {
    // sets = 1, associativity = 2, line_size = 4
    Cache cache(1, 2, 4, "LRU");

    EXPECT_FALSE(cache.access(0).hit); // Line 0 (Tag 0), Miss
    EXPECT_FALSE(cache.access(4).hit); // Line 1 (Tag 1), Miss

    // Access 0 to make it most recently used
    EXPECT_TRUE(cache.access(0).hit);  // Line 0 (Tag 0), Hit

    // Access 8, maps to Line 2 (Tag 2). Since set has capacity 2, it evicts the LRU.
    // LRU should be Line 1 (Tag 1) because Line 0 was just accessed.
    EXPECT_FALSE(cache.access(8).hit); // Line 2 (Tag 2), Miss + Eviction
    EXPECT_EQ(cache.stats().evictions, 1);

    // Verify Line 0 is still there (since it was MRU before 8 was accessed)
    EXPECT_TRUE(cache.access(0).hit);  // Line 0 (Tag 0), Hit

    // Now LRU is Line 2 (Tag 2). Line 0 is MRU.
    // Verify Line 1 was evicted earlier by accessing it and expecting a miss.
    EXPECT_FALSE(cache.access(4).hit); // Line 1 (Tag 1), Miss + Eviction (evicts Line 2)
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

    // Let's explicitly set writeback latency for clarity in calculations
    config.writeback_latency = 20;

    // Now let's hit L2 but miss L1
    // Evict 0 from L1 by accessing 8 (associativity is 2)
    // Currently L1 has 0 (dirty) and 4 (clean).
    // Load 8 (Evicts 0 from L1, occupies way 0. Misses L1, Misses L2). Since 0 is dirty, adds writeback latency.
    size_t load_8_expected_latency = expected_miss_latency + config.writeback_latency; // 125 + 20 = 145
    EXPECT_EQ(mem.load(8, val), load_8_expected_latency); // 145

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
    // L1 Dirty Eviction penalty component: (1 dirty eviction / 6 accesses) * 20 = 3.333333...
    // AMAT = 5 + (2/3 * 95) + (1/6 * 20) = 5 + 63.333333333333336 + 3.333333333333333 = 71.66666666666667

    EXPECT_DOUBLE_EQ(mem.compute_amat(), 5.0 + (2.0 / 3.0) * 95.0 + (1.0 / 6.0) * 20.0);
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
    EXPECT_FALSE(cache.access(0).hit); // A - miss
    EXPECT_FALSE(cache.access(4).hit); // B - miss

    // FIFO order is now A (oldest), then B (newest)

    // Access A again. In LRU this makes A the MRU. In FIFO, A is still the oldest.
    EXPECT_TRUE(cache.access(0).hit); // A - hit

    // Load C (Line 2). Should evict A, since it's the oldest loaded, despite being recently accessed.
    EXPECT_FALSE(cache.access(8).hit); // C - miss + eviction
    EXPECT_EQ(cache.stats().evictions, 1);

    // Verify A was evicted (so accessing it misses)
    EXPECT_FALSE(cache.access(0).hit); // A - miss + eviction (evicts B now)
    EXPECT_EQ(cache.stats().evictions, 2);
}

TEST(MemoryTest, RandomPolicyDeterminism) {
    // Two caches with same seed
    Cache cache1(1, 4, 4, "Random");
    Cache cache2(1, 4, 4, "Random");

    // Fill the 4 ways
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(cache1.access(i * 4).hit);
        EXPECT_FALSE(cache2.access(i * 4).hit);
    }

    // Now cause 10 evictions and verify they are identical
    for (size_t i = 4; i < 14; ++i) {
        EXPECT_FALSE(cache1.access(i * 4).hit);
        EXPECT_FALSE(cache2.access(i * 4).hit);

        // Since we can't inspect the cache ways directly,
        // their deterministic eviction choices imply the valid tags left inside are the same.
        // We will just verify they both have 1 eviction per step.
        EXPECT_EQ(cache1.stats().evictions, i - 3);
        EXPECT_EQ(cache2.stats().evictions, i - 3);
    }

    // Verify internal state by querying the same addresses. They should hit/miss identically.
    for (size_t i = 0; i < 14; ++i) {
        EXPECT_EQ(cache1.access(i * 4).hit, cache2.access(i * 4).hit);
    }
}

TEST(MemoryTest, CachePolicyIntegration) {
    // Just verifying the factory creates them without throwing
    EXPECT_NO_THROW(Cache(1, 2, 4, "LRU"));
    EXPECT_NO_THROW(Cache(1, 2, 4, "FIFO"));
    EXPECT_NO_THROW(Cache(1, 2, 4, "Random"));
    EXPECT_THROW(Cache(1, 2, 4, "UnknownPolicy"), std::invalid_argument);
}

// --- E7 Cache Writeback & Dirty Bit Tests ---
TEST(MemoryTest, DirtyBitOnWrite) {
    Cache cache(1, 2, 4);
    EXPECT_FALSE(cache.write(0).hit);
    EXPECT_EQ(cache.stats().misses, 1);

    EXPECT_TRUE(cache.access(0).hit); // read hit, remains dirty internally
}

TEST(MemoryTest, WriteMissIsWriteAllocate) {
    Cache cache(1, 1, 4);
    EXPECT_FALSE(cache.write(0).hit);
    auto res = cache.write(4);
    EXPECT_FALSE(res.hit);
    EXPECT_TRUE(res.eviction);
    EXPECT_TRUE(res.dirty_eviction);
    EXPECT_EQ(cache.stats().evictions, 1);
    EXPECT_EQ(cache.stats().dirty_evictions, 1);
}

TEST(MemoryTest, CleanEvictionCount) {
    Cache cache(1, 1, 4);
    EXPECT_FALSE(cache.access(0).hit); // clean
    auto res = cache.access(4);
    EXPECT_FALSE(res.hit);
    EXPECT_TRUE(res.eviction);
    EXPECT_FALSE(res.dirty_eviction);
    EXPECT_EQ(cache.stats().evictions, 1);
    EXPECT_EQ(cache.stats().dirty_evictions, 0);
}

TEST(MemoryTest, L1DirtyEvictionAddsLatency) {
    SharedMemory shared(65536);
    Cache l1(1, 1, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);

    MemoryAccessConfig config;
    config.l1_latency = 5;
    config.l2_latency = 20;
    config.global_memory_latency = 100;
    config.writeback_latency = 50;

    MemorySystem mem(shared, l1, l2, global, config);
    mem.store(0, 42);

    int val;
    size_t lat = mem.load(4, val);

    // Latency = l1_latency + (l2_latency + global) + writeback
    EXPECT_EQ(lat, 5 + (20 + 100) + 50);
}

TEST(MemoryTest, CleanEvictionAddsNoWritebackLatency) {
    SharedMemory shared(65536);
    Cache l1(1, 1, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);

    MemoryAccessConfig config;
    config.l1_latency = 5;
    config.l2_latency = 20;
    config.global_memory_latency = 100;
    config.writeback_latency = 50;

    MemorySystem mem(shared, l1, l2, global, config);
    int dummy;
    mem.load(0, dummy);

    int val;
    size_t lat = mem.load(4, val);

    // Latency = l1_latency + (l2_latency + global) without writeback
    EXPECT_EQ(lat, 5 + (20 + 100));
}

// --- E7 Banking Tests ---
TEST(MemoryTest, NoBankConflict) {
    SharedMemory sm(1024, 32);
    std::vector<size_t> addrs;
    for(int i=0; i<32; i++) addrs.push_back(i * 4);
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 0);
}

TEST(MemoryTest, All32SameBank) {
    SharedMemory sm(1024 * 32, 32);
    std::vector<size_t> addrs;
    for(int i=0; i<32; i++) addrs.push_back(i * 32 * 4);
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 31);
}

TEST(MemoryTest, BroadcastIsFree) {
    SharedMemory sm(1024, 32);
    std::vector<size_t> addrs;
    for(int i=0; i<32; i++) addrs.push_back(42 * 4);
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 0);
}

TEST(MemoryTest, MultipleBanksWithDifferentFanout) {
    SharedMemory sm(1024, 32);
    std::vector<size_t> addrs;
    for(int i=0; i<16; i++) addrs.push_back(i * 32 * 4); // 16 in bank 0
    for(int i=0; i<8; i++) addrs.push_back((i * 32 + 1) * 4); // 8 in bank 1
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 15);
}

// --- E7 Config & Integration Tests ---
TEST(MemoryTest, CustomConfigHierarchy) {
    MemoryAccessConfig config;
    config.l1_latency = 3;
    config.l2_latency = 12;
    config.global_memory_latency = 80;
    config.writeback_latency = 25;
    config.shared_memory_latency = 2;

    SharedMemory shared(1024, 16); // 16 banks
    Cache l1(4, 2, 8); // 4 sets, 2 ways, 8 line size
    Cache l2(8, 4, 8);
    GlobalMemory global(10000);

    MemorySystem mem(shared, l1, l2, global, config);

    int val;
    // L1 miss, L2 miss
    EXPECT_EQ(mem.load(0, val), 3 + 12 + 80);
    // L1 hit
    EXPECT_EQ(mem.load(0, val), 3);
    // L1 miss, L2 hit
    EXPECT_EQ(mem.load(8, val), 3 + 12 + 80);
    EXPECT_EQ(mem.load(8, val), 3);
}

// --- E7 Missing Test Coverage ---

/*
 * L2 dirty eviction is intentionally unreachable in E7 because L2 is populated
 * only through clean demand accesses and Rule 3 models L1 writeback as timing-only
 * without an actual L2 write.
 * Therefore, L2DirtyEvictionAddsLatency is intentionally not implemented.
 */

TEST(MemoryTest, CleanReadDoesNotDirty) {
    Cache cache(1, 1, 4);
    auto res = cache.access(0);
    EXPECT_FALSE(res.hit);

    // Evict it and see if it was dirty
    auto res2 = cache.access(4);
    EXPECT_TRUE(res2.eviction);
    EXPECT_FALSE(res2.dirty_eviction);
}

TEST(MemoryTest, DirtyEvictionCount) {
    Cache cache(1, 2, 4); // 1 set, 2 ways
    cache.write(0);
    cache.write(4);
    EXPECT_EQ(cache.stats().dirty_evictions, 0);

    // Overflow
    cache.write(8);
    EXPECT_EQ(cache.stats().dirty_evictions, 1);
    cache.write(12);
    EXPECT_EQ(cache.stats().dirty_evictions, 2);
}

TEST(MemoryTest, WritebackDoesNotBreakFunctionalCorrectness) {
    SharedMemory shared(65536);
    Cache l1(1, 1, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);
    MemoryAccessConfig config;
    MemorySystem mem(shared, l1, l2, global, config);

    mem.store(0, 999);

    // Evict line 0
    int dummy;
    mem.load(4, dummy);

    // Verify GlobalMemory still returns 999 (functionally correct)
    int val = 0;
    mem.load(0, val);
    EXPECT_EQ(val, 999);
}

TEST(MemoryTest, PartialWarpBankConflict) {
    SharedMemory sm(1024, 32);
    std::vector<size_t> addrs;
    // 5 threads accessing bank 0 (words 0, 32, 64, 96, 128)
    for(int i = 0; i < 5; i++) {
        addrs.push_back(i * 32 * 4);
    }
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 4);
}

TEST(MemoryTest, CustomLatencies) {
    SharedMemory shared(1024, 32);
    Cache l1(4, 2, 8);
    Cache l2(8, 4, 8);
    GlobalMemory global(10000);
    MemoryAccessConfig config;
    config.l1_latency = 7;
    config.l2_latency = 13;
    config.global_memory_latency = 55;
    config.writeback_latency = 33;
    MemorySystem mem(shared, l1, l2, global, config);

    int val;
    // L1 miss, L2 miss
    EXPECT_EQ(mem.load(0, val), 7 + 13 + 55);
    // L1 hit
    EXPECT_EQ(mem.load(0, val), 7);
}

TEST(MemoryTest, CustomBankCount) {
    SharedMemory sm(1024, 16); // 16 banks instead of 32
    std::vector<size_t> addrs;
    for(int i = 0; i < 4; i++) {
        addrs.push_back(i * 16 * 4); // all map to bank 0
    }
    EXPECT_EQ(sm.compute_bank_conflicts(addrs), 3);
}

TEST(MemoryTest, CustomGlobalMemorySize) {
    sim_sm::SystemConfig config;
    config.global_memory_size = 500000;
    
    // The GPU should successfully instantiate the global memory with the configured size.
    EXPECT_NO_THROW({ sim_sm::GPU gpu(config); });
}

TEST(MemoryTest, DefaultValuesPreserved) {
    sim_sm::SystemConfig config; // default constructor
    EXPECT_EQ(config.l1_sets, 4);
    EXPECT_EQ(config.l1_associativity, 4);
    EXPECT_EQ(config.l2_sets, 16);
    EXPECT_EQ(config.shared_memory_banks, 32);
    EXPECT_EQ(config.global_memory_latency, 100);
}

TEST(MemoryTest, SharedMemoryConflict) {
    SharedMemory shared(65536);
    Cache l1(1, 1, 4);
    Cache l2(1, 4, 4);
    GlobalMemory global(1000000);
    MemoryAccessConfig config;
    config.shared_memory_latency = 1;
    MemorySystem mem(shared, l1, l2, global, config);

    std::vector<size_t> addrs;
    std::vector<int> vals;
    for(int i = 0; i < 32; i++) {
        addrs.push_back(MemorySystem::SHARED_MEM_BASE + i * 32 * 4); // all bank 0
        vals.push_back(i);
    }
    auto res = mem.warp_store(addrs, vals);
    EXPECT_EQ(res.total_latency, 1 + 31);
}

TEST(MemoryTest, CustomL1Geometry) {
    Cache l1(2, 2, 8); // 2 sets, 2 ways, 8 bytes/line
    // Fill set 0 (addresses 0 and 16)
    l1.access(0);
    l1.access(16);
    EXPECT_EQ(l1.stats().evictions, 0);
    // Evict
    l1.access(32);
    EXPECT_EQ(l1.stats().evictions, 1);
}

TEST(MemoryTest, CustomL2Geometry) {
    Cache l2(4, 4, 16); // 4 sets, 4 ways, 16 bytes/line
    // Fill set 0
    l2.access(0);
    l2.access(64);
    l2.access(128);
    l2.access(192);
    EXPECT_EQ(l2.stats().evictions, 0);
    // Evict
    l2.access(256);
    EXPECT_EQ(l2.stats().evictions, 1);
}

} // namespace sim_sm
