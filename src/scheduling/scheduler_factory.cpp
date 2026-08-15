#include "scheduling/scheduler_factory.hpp"
#include "scheduling/round_robin_scheduler.hpp"
#include "scheduling/greedy_scheduler.hpp"
#include "scheduling/priority_scheduler.hpp"
#include "scheduling/oldest_first_scheduler.hpp"
#include "scheduling/gto_scheduler.hpp"
#include "scheduling/two_level_scheduler.hpp"
#include <stdexcept>

namespace sim_sm {

std::unique_ptr<WarpScheduler> SchedulerFactory::create_scheduler(const SystemConfig& config) {
    if (config.scheduler_policy == "RR") {
        return std::make_unique<RoundRobinScheduler>();
    } else if (config.scheduler_policy == "Greedy") {
        return std::make_unique<GreedyScheduler>();
    } else if (config.scheduler_policy == "Priority") {
        return std::make_unique<PriorityScheduler>();
    } else if (config.scheduler_policy == "OldestFirst") {
        return std::make_unique<OldestFirstScheduler>();
    } else if (config.scheduler_policy == "GTO") {
        return std::make_unique<GTOScheduler>();
    } else if (config.scheduler_policy == "TwoLevel") {
        return std::make_unique<TwoLevelScheduler>(config.active_set_size);
    } else {
        throw std::invalid_argument("Unknown scheduler policy: " + config.scheduler_policy);
    }
}

} // namespace sim_sm
