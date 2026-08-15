#include "execution/instruction_executor.hpp"
#include <stdexcept>
#include <unordered_map>

namespace sim_sm {

ExecutionResult InstructionExecutor::execute(const Instruction& inst, Warp& warp, MemorySystem& memory) {
    ExecutionStatus overall_status = ExecutionStatus::Completed;
    size_t max_latency = 1; // Default latency
    size_t memory_transactions = 0;
    std::string memory_space = "";
    size_t write_conflict_stalls = 0;
    size_t bank_conflicts = 0;
    size_t dirty_evictions = 0;

    size_t warp_pc = warp.get_warp_pc();
    std::bitset<32> active_mask = warp.get_active_mask();

    auto update_shadow_pcs = [&](size_t next_pc) {
        for (size_t i = 0; i < warp.get_threads().size(); ++i) {
            if (active_mask.test(i)) {
                warp.get_threads()[i].set_pc(next_pc);
            }
        }
    };

    switch (inst.opcode) {
        case Opcode::ADD: {
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 + val2);
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::SUB: {
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 - val2);
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::MUL: {
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.registers().write(inst.dst, val1 * val2);
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::MOV: {
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                if (inst.src1 == -1) {
                    thread.registers().write(inst.dst, inst.immediate);
                } else {
                    int val = thread.registers().read(inst.src1);
                    thread.registers().write(inst.dst, val);
                }
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::LOAD: {
            std::vector<size_t> addresses;
            std::vector<size_t> active_indices;
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                size_t base = (inst.src1 != -1) ? thread.registers().read(inst.src1) : 0;
                addresses.push_back(base + inst.immediate);
                active_indices.push_back(i);
            }

            if (!addresses.empty()) {
                if (addresses[0] >= MemorySystem::SHARED_MEM_BASE) {
                    memory_space = "SHARED";
                } else {
                    memory_space = "GLOBAL";
                }
                std::vector<int> out_values;
                WarpMemoryResult res = memory.warp_load(addresses, out_values);
                max_latency = res.total_latency;
                memory_transactions = res.num_transactions;
                bank_conflicts = res.bank_conflict_stalls;
                dirty_evictions = res.dirty_eviction_writebacks;

                for (size_t k = 0; k < active_indices.size(); ++k) {
                    size_t thread_idx = active_indices[k];
                    warp.get_threads()[thread_idx].registers().write(inst.dst, out_values[k]);
                }
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::STORE: {
            std::vector<size_t> addresses;
            std::vector<int> values;
            std::vector<size_t> active_indices;
            std::unordered_map<size_t, size_t> address_counts;

            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                size_t base = (inst.src2 != -1) ? thread.registers().read(inst.src2) : 0;
                size_t addr = base + inst.immediate;
                addresses.push_back(addr);
                values.push_back(thread.registers().read(inst.src1));
                active_indices.push_back(i);
                address_counts[addr]++;
            }

            size_t max_collisions = 0;
            for (const auto& pair : address_counts) {
                if (pair.second > max_collisions) max_collisions = pair.second;
            }

            if (!addresses.empty()) {
                if (addresses[0] >= MemorySystem::SHARED_MEM_BASE) {
                    memory_space = "SHARED";
                } else {
                    memory_space = "GLOBAL";
                }
                WarpMemoryResult res = memory.warp_store(addresses, values);
                max_latency = res.total_latency;
                memory_transactions = res.num_transactions;
                bank_conflicts = res.bank_conflict_stalls;
                dirty_evictions = res.dirty_eviction_writebacks;

                if (max_collisions > 1) {
                    write_conflict_stalls = max_collisions - 1;
                    max_latency += write_conflict_stalls;
                }
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::ATOMIC_ADD: {
            std::vector<size_t> addresses;
            std::vector<int> values;
            std::vector<size_t> active_indices;
            std::unordered_map<size_t, size_t> address_counts;

            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                size_t base = (inst.src2 != -1) ? thread.registers().read(inst.src2) : 0;
                size_t addr = base + inst.immediate;
                addresses.push_back(addr);
                values.push_back(thread.registers().read(inst.src1));
                active_indices.push_back(i);
                address_counts[addr]++;
            }

            size_t max_collisions = 0;
            for (const auto& pair : address_counts) {
                if (pair.second > max_collisions) max_collisions = pair.second;
            }

            if (!addresses.empty()) {
                if (addresses[0] >= MemorySystem::SHARED_MEM_BASE) {
                    memory_space = "SHARED";
                } else {
                    memory_space = "GLOBAL";
                }
                WarpMemoryResult res = memory.warp_atomic_add(addresses, values);
                max_latency = res.total_latency;
                memory_transactions = res.num_transactions;
                bank_conflicts = res.bank_conflict_stalls;
                dirty_evictions = res.dirty_eviction_writebacks;

                if (max_collisions > 1) {
                    write_conflict_stalls = max_collisions - 1;
                    max_latency += write_conflict_stalls;
                }
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::CMP: {
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (!active_mask.test(i)) continue;
                auto& thread = warp.get_threads()[i];
                int val1 = thread.registers().read(inst.src1);
                int val2 = thread.registers().read(inst.src2);
                thread.set_predicate(val1 == val2);
            }
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::BRANCH: {
            std::bitset<32> taken_mask = 0;
            for (size_t i = 0; i < warp.get_threads().size(); ++i) {
                if (active_mask.test(i) && warp.get_threads()[i].predicate()) {
                    taken_mask.set(i);
                }
            }

            if (taken_mask.none()) {
                // All active threads fall through
                warp.set_warp_pc(warp_pc + 1);
            } else if (taken_mask == active_mask) {
                // All active threads take branch
                warp.set_warp_pc(warp_pc + inst.immediate);
                overall_status = ExecutionStatus::BranchTaken;
            } else {
                // Divergent branch
                std::bitset<32> not_taken_mask = active_mask & ~taken_mask;
                
                // Push deferred path (not taken)
                SIMTStackEntry deferred;
                deferred.active_mask = not_taken_mask;
                deferred.target_pc = warp_pc + 1;
                deferred.reconvergence_pc = warp.get_reconvergence_pc();
                warp.push_simt_stack(deferred);

                // Execute taken path now
                warp.set_active_mask(taken_mask);
                warp.set_warp_pc(warp_pc + inst.immediate);
                overall_status = ExecutionStatus::BranchTaken;
            }
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::SSY: {
            // Push reconvergence token: target_pc is the SYNC instruction, reconvergence_pc is the PREVIOUS reconvergence PC
            SIMTStackEntry rec_token;
            rec_token.active_mask = active_mask;
            rec_token.target_pc = warp_pc + inst.immediate;
            rec_token.reconvergence_pc = warp.get_reconvergence_pc();
            warp.push_simt_stack(rec_token);

            warp.set_reconvergence_pc(warp_pc + inst.immediate);
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        case Opcode::SYNC: {
            SIMTStackEntry next_path;
            if (warp.pop_simt_stack(next_path)) {
                if (next_path.target_pc == warp_pc) {
                    // This is the reconvergence token!
                    warp.set_active_mask(next_path.active_mask);
                    warp.set_reconvergence_pc(next_path.reconvergence_pc);
                    warp.set_warp_pc(warp_pc + 1); // Advance past SYNC
                } else {
                    // This is a deferred path
                    warp.set_active_mask(next_path.active_mask);
                    warp.set_warp_pc(next_path.target_pc);
                    warp.set_reconvergence_pc(next_path.reconvergence_pc);
                }
                update_shadow_pcs(warp.get_warp_pc());
            } else {
                throw std::runtime_error("SYNC executed with empty SIMT stack");
            }
            break;
        }
        case Opcode::BARRIER: {
            size_t active_count = active_mask.count();
            if (active_count != warp.get_threads().size()) {
                // Assuming block size == warp size for simplicity in this check, 
                // but actually, all threads in the warp must participate in the barrier.
                throw std::runtime_error("Divergent barrier within warp");
            }
            overall_status = ExecutionStatus::BarrierReached;
            warp.set_warp_pc(warp_pc + 1);
            update_shadow_pcs(warp.get_warp_pc());
            break;
        }
        default:
            throw std::runtime_error("Unknown opcode");
    }

    return {overall_status, max_latency, memory_transactions, memory_space, write_conflict_stalls, bank_conflicts, dirty_evictions};
}

} // namespace sim_sm
