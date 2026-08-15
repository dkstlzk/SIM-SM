# SIM-SM Architectural Analysis

## Executive Summary

**Baseline configuration:** Config_0
- Scheduler: RR
- L1 Sets: 4

**Best configuration:** Config_8
- Scheduler: OldestFirst
- L1 Sets: 16
- Max Registers: 65536
- Max Shared Memory: 65536
- Max Warps: 64

**Speedup over baseline:** 1.04x

**Primary bottleneck (Best Config):** Memory Latency

**Confidence:** High

## Configuration Comparison

| Config | Scheduler | L1 Sets | Cycles | IPC | AMAT | Occupancy | Fairness | Speedup |
|--------|-----------|---------|--------|-----|------|-----------|----------|---------|
| Config_0 (Base) | RR | 4 | 42936 | 0.60 | 122.00 | 0.50 | 0.25 | 1.00x |
| Config_1 | RR | 8 | 43022 | 0.60 | 123.70 | 0.50 | 0.25 | 1.00x |
| Config_2 | RR | 16 | 42496 | 0.60 | 124.46 | 0.50 | 0.25 | 1.01x |
| Config_3 | GTO | 4 | 42342 | 0.60 | 128.04 | 0.50 | 0.25 | 1.01x |
| Config_4 | GTO | 8 | 42729 | 0.60 | 126.97 | 0.50 | 0.25 | 1.00x |
| Config_5 | GTO | 16 | 41862 | 0.61 | 130.52 | 0.50 | 0.25 | 1.03x |
| Config_6 | OldestFirst | 4 | 42280 | 0.61 | 127.52 | 0.50 | 0.25 | 1.02x |
| Config_7 | OldestFirst | 8 | 42398 | 0.60 | 126.39 | 0.50 | 0.25 | 1.01x |
| Config_8 | OldestFirst | 16 | 41422 | 0.62 | 128.53 | 0.50 | 0.25 | 1.04x |
| Config_9 | TwoLevel | 4 | 166598 | 0.15 | 131.56 | 0.50 | 0.25 | 0.26x |
| Config_10 | TwoLevel | 8 | 164118 | 0.16 | 131.42 | 0.50 | 0.25 | 0.26x |
| Config_11 | TwoLevel | 16 | 164058 | 0.16 | 131.57 | 0.50 | 0.25 | 0.26x |

## Bottleneck Diagnosis

### Config_0 (Baseline)
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (121.998760)
- Low L1 hit rate (0.044075)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles

### Config_1
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (123.700410)
- Low L1 hit rate (0.030983)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: +0.2%
- IPC: -0.2%
- AMAT: +1.4%
- Occupancy: +0.0
- Fairness: +0.0

### Config_2
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (124.457079)
- Low L1 hit rate (0.042140)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -1.0%
- IPC: +1.0%
- AMAT: +2.0%
- Occupancy: +0.0
- Fairness: +0.0

### Config_3
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (128.040569)
- Low L1 hit rate (0.003121)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -1.4%
- IPC: +1.4%
- AMAT: +5.0%
- Occupancy: +0.0
- Fairness: +0.0

### Config_4
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (126.970973)
- Low L1 hit rate (0.012697)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -0.5%
- IPC: +0.5%
- AMAT: +4.1%
- Occupancy: +0.0
- Fairness: +0.0

### Config_5
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (130.517698)
- Low L1 hit rate (0.003249)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -2.5%
- IPC: +2.6%
- AMAT: +7.0%
- Occupancy: +0.0
- Fairness: +0.0

### Config_6
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (127.520947)
- Low L1 hit rate (0.004051)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -1.5%
- IPC: +1.6%
- AMAT: +4.5%
- Occupancy: +0.0
- Fairness: +0.0

### Config_7
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (126.394494)
- Low L1 hit rate (0.014439)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -1.3%
- IPC: +1.3%
- AMAT: +3.6%
- Occupancy: +0.0
- Fairness: +0.0

### Config_8
**Primary:** Memory Latency
**Secondary:** Compute
**Evidence:**
- High AMAT (128.525137)
- Low L1 hit rate (0.010698)
- Uncoalesced accesses driving high memory transactions
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: -3.5%
- IPC: +3.7%
- AMAT: +5.3%
- Occupancy: +0.0
- Fairness: +0.0

### Config_9
**Primary:** Memory Latency
**Secondary:** Scheduler Inefficiency, Compute
**Evidence:**
- High AMAT (131.557798)
- Low L1 hit rate (0.000203)
- Uncoalesced accesses driving high memory transactions
- Scheduler Inefficiency -> Starvation events detected (81)
- Scheduler Inefficiency -> High max warp wait cycles (124604)
- Scheduler Inefficiency -> Low fairness (0.250000)
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: +288.0%
- IPC: -74.2%
- AMAT: +7.8%
- Occupancy: +0.0
- Fairness: +0.0

### Config_10
**Primary:** Memory Latency
**Secondary:** Scheduler Inefficiency, Compute
**Evidence:**
- High AMAT (131.424205)
- Low L1 hit rate (0.000032)
- Uncoalesced accesses driving high memory transactions
- Scheduler Inefficiency -> Starvation events detected (85)
- Scheduler Inefficiency -> High max warp wait cycles (123027)
- Scheduler Inefficiency -> Low fairness (0.250000)
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: +282.2%
- IPC: -73.8%
- AMAT: +7.7%
- Occupancy: +0.0
- Fairness: +0.0

### Config_11
**Primary:** Memory Latency
**Secondary:** Scheduler Inefficiency, Compute
**Evidence:**
- High AMAT (131.566347)
- Low L1 hit rate (0.000182)
- Uncoalesced accesses driving high memory transactions
- Scheduler Inefficiency -> Starvation events detected (80)
- Scheduler Inefficiency -> High max warp wait cycles (122987)
- Scheduler Inefficiency -> Low fairness (0.250000)
- Compute -> Low NoReadyWarpCycles
**Deltas vs Baseline:**
- Cycles: +282.1%
- IPC: -73.8%
- AMAT: +7.8%
- Occupancy: +0.0
- Fairness: +0.0

## Parameter Sensitivity

### Average Speedup by Scheduler Policy
- **GTO**: 1.01x
- **OldestFirst**: 1.02x
- **RR**: 1.00x
- **TwoLevel**: 0.26x

### Average Speedup by L1 Sets
- **4 sets**: 0.82x
- **8 sets**: 0.82x
- **16 sets**: 0.83x

*Note: These averages are descriptive sensitivity summaries across tested configurations and may include interactions between simultaneously swept parameters. Causal bottleneck conclusions are based exclusively on matched-configuration isolation.*

## Architectural Recommendations

1. The best configuration is still memory latency bound. Consider increasing `l1_sets`, `l2_sets`, or changing cache policies.
   *(Future capability note: hardware prefetching is not currently supported in SIM-SM but would further mitigate this.)*
