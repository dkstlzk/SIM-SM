#pragma once

#include "architecture/register_file.hpp"
#include <cstddef>

namespace sim_sm {

class Thread {
public:
    Thread(size_t global_id, size_t block_id, size_t local_id, size_t warp_id, size_t lane_id);

    size_t get_global_id() const;
    size_t get_block_id() const;
    size_t get_local_id() const;
    size_t get_warp_id() const;
    size_t get_lane_id() const;

    // Execution state
    RegisterFile& registers();
    const RegisterFile& registers() const;

    int pc() const;
    void set_pc(int pc);

    bool predicate() const;
    void set_predicate(bool value);

private:
    // Identity
    size_t global_id_;
    size_t block_id_;
    size_t local_id_;
    size_t warp_id_;
    size_t lane_id_;

    // Execution state
    RegisterFile registers_;
    int pc_;
    bool predicate_;
};

} // namespace sim_sm
