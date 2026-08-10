# SIM-SM: CUDA Streaming Multiprocessor Simulator

SIM-SM is a cycle-level educational architectural simulator for a CUDA-like Streaming Multiprocessor (SM). It models a deliberately constrained subset of GPU execution, including warp scheduling, memory hierarchy, memory coalescing, occupancy, forward divergence, and block-level synchronization.

## Week 1 Features & Capabilities

- **GPU Hierarchy**: Hierarchical software model of GPU, SM, Thread Blocks, Warps, and Threads.
- **Instruction Set & Execution**: A custom lightweight ISA supporting standard ALU operations, branching, loads, stores, and barriers.
- **Warp Scheduling**: Cycle-level SM tick mechanism with configurable schedulers (Round-Robin, Greedy, Priority).
- **Memory Subsystem**: L1/L2 caches, Shared Memory, Global Memory, pluggable cache replacement policies (LRU, FIFO, Random), and access latencies (AMAT).
- **Memory Coalescing**: **Warp-level global memory coalescing modeled using cache-line transactions.**
- **Multi-Wave Dispatch & Occupancy**: Accurate SM resource tracking (Registers, Shared Memory, Threads) allowing for multiple resident blocks, dynamically dispatching based on occupancy.
- **Constrained Divergence & Synchronization**: 
  - Forward-divergence paths correctly isolate non-participating threads.
  - Block-scoped `BARRIER` synchronization modeled at the SM level.
  - Strict malformed barrier and divergence exception detection.
- **Comprehensive Benchmarks**: Built-in CLI for executing complex kernels (e.g., Vector Add, Cache Stress, Matrix Multiply) and logging occupancy and IPC metrics to CSVs.

## Test Suite

The simulator is rigorously verified via GoogleTest suites covering component-level edge cases and complete system-level integration.

Currently running **47/47** passing `ctest` regression tests:

- **Architecture Tests**: Verifies hierarchy, partial warps, block occupancy, and structural barrier invariants.
- **Execution Tests**: Instruction level testing of math, load/store semantics, branching, and boundaries.
- **Scheduling Tests**: Unit/integration tests for Greedy, Round-Robin, and Priority scheduler policies.
- **Memory Tests**: Checks cache associativity, LRU/FIFO/Random eviction, hit/miss ratios, and multi-tier access times.
- **Coalescing Tests**: Verifies transaction merging across sequential, strided, and scattered memory access patterns.
- **Divergence & Synchronization Tests**: Enforces proper `BARRIER` stalls, multi-warp release invariants, and malformed barrier detection for mismatched thread PCs within warps.

## Building and Testing

Requirements: `CMake 3.14+`, a C++17 compliant compiler.

```bash
mkdir build && cd build
cmake ..
make -j4

# Run all tests
ctest --output-on-failure
```

## Running Benchmarks and Tracing

SIM-SM includes a command-line interface to execute internal benchmark scenarios (Cache Stress, Scheduler sweep, Coalescing Sweep, Vector Add).

```bash
./build/gpu-sim --config configs/small_gpu.json --benchmark all
```
*Results are output as CSV files in the `results/` directory.*

### Debug / Trace Mode (Extension 3)

You can capture a per-cycle execution trace of the GPU model to observe scheduling, memory, and cache events.

```bash
# Basic tracing to standard output
./build/gpu-sim --config configs/small_gpu.json --debug --benchmark basic

# Specify trace level (1: Scheduler, 2: Memory, 4: Cache, 7: All)
./build/gpu-sim --config configs/small_gpu.json --debug --trace-level 1 --benchmark basic

# Redirect trace to a file
./build/gpu-sim --config configs/small_gpu.json --debug --trace-file my_trace.log --benchmark basic
```

## Future Work (Week 2+)

- Reconvergence stack (SIMT mask) for arbitrary divergence topologies.
- Bank conflict modeling in Shared Memory.
- True multi-SM interconnect modeling and L2 slice partitioning.
