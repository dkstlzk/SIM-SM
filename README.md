# SIM-SM: CUDA-like GPU Architecture Simulator

SIM-SM is a cycle-level educational CUDA-like GPU architecture simulator implemented in C++17. It models SM/warp execution, configurable warp scheduling, hierarchical memory, coalescing, occupancy, constrained divergence and synchronization, multiple benchmark workloads, and architectural performance analysis with causal bottleneck validation.

## Current capabilities:
- GPU/SM/warp/thread hierarchy
- Custom ISA and execution
- RR / Greedy / Priority / GTO / OldestFirst / TwoLevel scheduling
- L1/L2/shared/global memory
- Cache replacement policies (LRU, FIFO, Random)
- Memory coalescing
- Divergence and barriers
- Occupancy/resource modeling
- Performance counters
- Vector Add
- Memcpy
- Reduction
- Histogram
- Matrix Multiply
- Architectural parameter sweeps
- Bottleneck diagnosis
- Markdown architectural reports

## Test status
The simulator is rigorously verified via GoogleTest suites covering component-level edge cases and complete system-level integration.

Currently running **98/98** passing `ctest` regression tests:

- **Architecture Tests**: Verifies hierarchy, partial warps, block occupancy, and structural barrier invariants.
- **Execution Tests**: Instruction level testing of math, load/store semantics, branching, and boundaries.
- **Scheduling Tests**: Unit/integration tests for Greedy, Round-Robin, Priority, GTO, OldestFirst, and TwoLevel scheduler policies.
- **Memory Tests**: Checks cache associativity, LRU/FIFO/Random eviction, hit/miss ratios, writeback, and multi-tier access times.
- **Coalescing Tests**: Verifies transaction merging across sequential, strided, and scattered memory access patterns.
- **Divergence & Synchronization Tests**: Enforces proper `BARRIER` stalls, multi-warp release invariants, and malformed barrier detection for mismatched thread PCs within warps.
- **Occupancy Tests**: Validates max threads, blocks, shared memory, and registers limits.
- **Analysis Tests**: Verifies bottleneck diagnosis, delta calculations, Cartesian sweeps, and causal evidence isolation.
- **Benchmark Tests**: Validates full correctness of workloads, including a dedicated CPU-validation step for GEMM (Matrix Multiply).

## Building and Testing

Requirements: `CMake 3.14+`, a C++17 compliant compiler.

```bash
mkdir build && cd build
cmake ..
make -j8

# Run all tests
ctest --output-on-failure
```

## Running Benchmarks and Analysis

SIM-SM includes a command-line interface to execute internal benchmark scenarios and architectural parameter sweeps.

### Running Benchmarks
Run a specific benchmark using the `--benchmark` flag:

```bash
./build/gpu-sim --config configs/small_gpu.json --benchmark matrix_multiply
./build/gpu-sim --config configs/small_gpu.json --benchmark all
./build/gpu-sim --config configs/small_gpu.json --benchmark basic
./build/gpu-sim --config configs/small_gpu.json --benchmark priority
./build/gpu-sim --config configs/small_gpu.json --benchmark memcpy
./build/gpu-sim --config configs/small_gpu.json --benchmark reduction
./build/gpu-sim --config configs/small_gpu.json --benchmark histogram
```
*Results are output as CSV files in the `results/` directory.*

### Running Architectural Analysis
Run the automated Cartesian parameter sweep and bottleneck diagnosis:

```bash
./build/gpu-sim --config configs/small_gpu.json --analyze
```
*A detailed Markdown report is generated in `results/architectural_analysis_report.md`.*

### Debug / Trace Mode

You can capture a per-cycle execution trace of the GPU model to observe scheduling, memory, and cache events.

```bash
# Basic tracing to standard output
./build/gpu-sim --config configs/small_gpu.json --debug --benchmark basic

# Specify trace level (1: Scheduler, 2: Memory, 4: Cache, 7: All)
./build/gpu-sim --config configs/small_gpu.json --debug --trace-level 1 --benchmark basic

# Redirect trace to a file
./build/gpu-sim --config configs/small_gpu.json --debug --trace-file my_trace.log --benchmark basic
```
