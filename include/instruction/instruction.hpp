#pragma once

namespace sim_sm {

enum class Opcode {
    ADD,
    SUB,
    MUL,
    MOV,
    LOAD,
    STORE,
    CMP,
    BRANCH,
    BARRIER,
    ATOMIC_ADD,
    SSY,
    SYNC
};

// Operand conventions for the core ISA and post-Week-1 additions:
//
// MOV    dst, src1
// MOV    dst, immediate   (encoded with src1 = -1)
//
// ADD    dst, src1, src2
// SUB    dst, src1, src2
// MUL    dst, src1, src2
//
// LOAD   dst, [immediate]
// STORE  [immediate], src1
//
// CMP    src1, src2
// BRANCH immediate        (if predicate is true, PC += immediate; otherwise PC += 1)
// BARRIER

struct Instruction {
    Opcode opcode;
    int dst;
    int src1;
    int src2;
    int immediate;
};

} // namespace sim_sm
