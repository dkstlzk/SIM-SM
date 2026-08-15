#pragma once

#include "scheduling/warp_scheduler.hpp"
#include "runtime/config.hpp"
#include <memory>
#include <string>

namespace sim_sm {

class SchedulerFactory {
public:
    static std::unique_ptr<WarpScheduler> create_scheduler(const SystemConfig& config);
};

} // namespace sim_sm
