#include "runtime/trace_logger.hpp"
#include <stdexcept>

namespace sim_sm {

TraceLogger::TraceLogger(TraceLevel level, const std::string& output_file)
    : level_(level) {
    if (!output_file.empty()) {
        file_stream_ = std::make_unique<std::ofstream>(output_file);
        if (!file_stream_->is_open()) {
            throw std::runtime_error("Could not open trace output file: " + output_file);
        }
        out_ = file_stream_.get();
    } else {
        out_ = &std::cout;
    }
}

TraceLogger::~TraceLogger() {
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->close();
    }
}

bool TraceLogger::is_enabled(TraceLevel level) const {
    return (level_ & level) != TraceLevel::None;
}

void TraceLogger::log_scheduler_event(size_t sm_id, size_t cycle, const std::string& scheduler_name, size_t selected_warp_id, const std::vector<Warp>& warps) {
    if (!is_enabled(TraceLevel::Scheduler)) return;
    
    *out_ << "Cycle " << cycle << ":\n";
    *out_ << "SM" << sm_id << ":\n";
    *out_ << "  Scheduler: " << scheduler_name << "\n";
    *out_ << "  Selected Warp: " << selected_warp_id << "\n";
    for (const auto& w : warps) {
        *out_ << "  Warp " << w.get_warp_id() << ": ";
        switch (w.get_state()) {
            case WarpState::Ready: *out_ << "READY\n"; break;
            case WarpState::StalledAtBarrier: *out_ << "WAITING_BARRIER\n"; break;
            case WarpState::Stalled: *out_ << "STALLED\n"; break;
            case WarpState::Completed: *out_ << "COMPLETED\n"; break;
        }
    }
}

void TraceLogger::log_memory_event(size_t sm_id, size_t cycle, size_t warp_id, const Instruction& inst, size_t transactions, const std::string& memory_space) {
    if (!is_enabled(TraceLevel::Memory)) return;
    *out_ << "Cycle " << cycle << ":\n";
    *out_ << "SM" << sm_id << ":\n";
    *out_ << "  Memory Event:\n";
    *out_ << "    WarpID=" << warp_id << "\n";
    std::string op_str = (inst.opcode == Opcode::LOAD) ? "LOAD" : "STORE";
    *out_ << "    Opcode=" << op_str << " addr=0x" << std::hex << inst.immediate << std::dec << "\n";
    *out_ << "    Transactions=" << transactions << "\n";
    *out_ << "    MemorySpace=" << memory_space << "\n";
}

void TraceLogger::log_cache_event(size_t cycle, const std::string& cache_name, size_t set, size_t way, bool hit) {
    if (!is_enabled(TraceLevel::Cache)) return;
    
    *out_ << "Cycle " << cycle << ":\n";
    *out_ << cache_name << ":\n";
    *out_ << "  Cache Event:\n";
    *out_ << "    SET=" << set << "\n";
    *out_ << "    WAY=" << way << "\n";
    *out_ << "    Result=" << (hit ? "HIT" : "MISS") << "\n";
}

} // namespace sim_sm
