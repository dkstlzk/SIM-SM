#pragma once

#include "analysis/analysis_types.hpp"
#include <string>
#include <vector>

namespace sim_sm {

class ReportGenerator {
public:
    static std::string generate_markdown_report(const std::vector<AnalysisResult>& results, const std::string& title = "SIM-SM Architectural Analysis");
};

} // namespace sim_sm
