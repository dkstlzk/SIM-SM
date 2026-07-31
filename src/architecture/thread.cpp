#include "architecture/thread.hpp"

namespace sim_sm {

Thread::Thread(size_t global_id, size_t block_id, size_t local_id, size_t warp_id, size_t lane_id)
    : global_id_(global_id)
    , block_id_(block_id)
    , local_id_(local_id)
    , warp_id_(warp_id)
    , lane_id_(lane_id)
    , registers_(32) // default 32 registers per thread
    , pc_(0)
    , predicate_(false)
{}

size_t Thread::get_global_id() const { return global_id_; }
size_t Thread::get_block_id() const { return block_id_; }
size_t Thread::get_local_id() const { return local_id_; }
size_t Thread::get_warp_id() const { return warp_id_; }
size_t Thread::get_lane_id() const { return lane_id_; }

RegisterFile& Thread::registers() { return registers_; }
const RegisterFile& Thread::registers() const { return registers_; }

int Thread::pc() const { return pc_; }
void Thread::set_pc(int pc) { pc_ = pc; }

bool Thread::predicate() const { return predicate_; }
void Thread::set_predicate(bool value) { predicate_ = value; }

} // namespace sim_sm
