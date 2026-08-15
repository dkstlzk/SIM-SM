# SIM-SM — Streaming Multiprocessor Simulator

SIM-SM (Streaming Multiprocessor Simulator) is a cycle-level, educational CUDA-like GPU architecture simulator implemented in C++17. 

## What this project demonstrates
This project is an architectural modeling and performance-analysis framework, not merely an instruction interpreter. It demonstrates:
- Architectural modeling and performance experimentation
- Configurable warp scheduling policies
- Memory-system modeling and coalescing
- SIMT execution with constrained divergence
- Correctness testing and regression validation
- Quantitative performance analysis and Cartesian parameter sweeps
- Causal architectural reasoning and bottleneck diagnosis

## Current Project Status
SIM-SM is currently at a stable, tested milestone suitable for use as an architectural/interview project. The core simulator, benchmark suite, testing infrastructure, and architectural analysis framework are implemented and validated.

## Scope and Limitations
SIM-SM is an educational architectural simulator and intentionally abstracts many details of real GPUs. It does NOT model:
- Full CUDA/PTX compatibility or compilation
- A specific NVIDIA GPU's exact microarchitecture
- Complete hardware timing behavior
- All real GPU pipelines, microarchitectural structures, or implementation-specific hardware behavior

Additionally, the current SIMT control-flow model is structured and constrained (with push/pop reconvergence), rather than a complete, arbitrary CUDA control-flow implementation. These constraints are deliberate project-scope boundaries designed to keep the simulator legible and modifiable.

## Current Capabilities

### Architecture & Execution
- GPU/SM/thread-block/warp/thread hierarchy
- Custom ISA and instruction execution
- SIMT execution
- Warp divergence with reconvergence support
- Structured synchronization/barrier modeling
- Occupancy and resource modeling

### Scheduling
Six warp-scheduling policies are natively implemented:
- Round-Robin (RR)
- Greedy
- Priority
- GTO
- OldestFirst
- TwoLevel

### Memory System
- Registers
- Shared memory
- L1 cache
- L2 cache
- Global memory
- Memory coalescing
- Cache replacement policies: LRU, FIFO, Random
- Writeback / dirty eviction behavior

### Instrumentation & Analysis
- Performance counters
- Scheduler starvation / warp wait metrics
- Fairness metrics
- Architectural parameter sweeps
- Bottleneck diagnosis
- Baseline-relative performance deltas
- Causal evidence isolation using matched configurations
- Markdown architectural analysis reports

### Benchmarks
The repository contains the following natively implemented and validated benchmarks:
- Vector Add
- Memcpy
- Reduction
- Histogram
- Matrix Multiply / GEMM

The `priority` benchmark runs the same workload under priority-based scheduling for scheduler comparison.

## Architectural Analysis Framework
SIM-SM includes an automated architectural analysis engine that goes beyond simple metric collection. It:
- performs Cartesian configuration sweeps
- compares configurations against a baseline
- computes cycle/IPC/AMAT/occupancy/fairness deltas
- diagnoses architectural bottlenecks
- reports memory latency, compute, occupancy, scheduler inefficiency, and bank-conflict signals where applicable
- uses matched-configuration isolation when making causal claims about swept architectural parameters
- generates results/architectural_analysis_report.md

Note that descriptive averages across a sweep are not automatically causal; causal conclusions require matched-configuration isolation. The analyzer distinguishes descriptive sweep-level trends from matched-configuration evidence used for causal conclusions.

Run the automated Cartesian parameter sweep and bottleneck diagnosis:
```bash
./build/gpu-sim --config configs/small_gpu.json --analyze
```

## Test Status
At the current freeze milestone: 98/98 GoogleTest cases passing.

The test suite covers:
- **Architecture Tests**: Verifies hierarchy, partial warps, block occupancy, and structural barrier invariants.
- **Execution Tests**: Instruction level testing of math, load/store semantics, branching, and boundaries.
- **Scheduling Tests**: Unit/integration tests for Greedy, Round-Robin, Priority, GTO, OldestFirst, and TwoLevel scheduler policies.
- **Memory Tests**: Checks cache associativity, LRU/FIFO/Random eviction, hit/miss ratios, writeback, and multi-tier access times.
- **Coalescing Tests**: Verifies transaction merging across sequential, strided, and scattered memory access patterns.
- **Divergence & Synchronization Tests**: Enforces structured/constrained `BARRIER` stalls, multi-warp release invariants, and malformed barrier detection for mismatched thread PCs within warps.
- **Occupancy Tests**: Validates max threads, blocks, shared memory, and registers limits.
- **Analysis Tests**: Verifies bottleneck diagnosis, delta calculations, Cartesian sweeps, and causal evidence isolation.
- **Benchmark Tests**: Validates full correctness of workloads, including dedicated CPU-reference validation for GEMM / Matrix Multiply.

## Building and Testing
Requirements: `CMake 3.14+`, a C++17 compliant compiler.

```bash
mkdir build
cd build
cmake ..
cmake --build . -j8
ctest --output-on-failure
```

## Running Benchmarks
SIM-SM includes a command-line interface to execute internal benchmark scenarios. Matrix Multiply and other benchmarks are fully implemented and validated. The benchmark names correspond directly to the CLI interface. Results are written under `results/`.

```bash
./build/gpu-sim --config configs/small_gpu.json --benchmark matrix_multiply
./build/gpu-sim --config configs/small_gpu.json --benchmark all
./build/gpu-sim --config configs/small_gpu.json --benchmark basic
./build/gpu-sim --config configs/small_gpu.json --benchmark priority
./build/gpu-sim --config configs/small_gpu.json --benchmark memcpy
./build/gpu-sim --config configs/small_gpu.json --benchmark reduction
./build/gpu-sim --config configs/small_gpu.json --benchmark histogram
```

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

## Future Extensions / Deferred Work
Possible next extensions include:

1. CPU–GPU transfer / PCIe-style transfer cost modeling
   - Model host-device transfer latency and bandwidth effects.
2. Python-based visualization / plotting pipeline
   - Generate plots for architectural sweeps, bottlenecks, cache behavior, scheduler behavior, and performance trends.
3. External profiling / perf integration
   - Integrate host-side profiling tools such as Linux `perf` to study simulator overhead and execution characteristics.

These are future extensions, not current capabilities.
