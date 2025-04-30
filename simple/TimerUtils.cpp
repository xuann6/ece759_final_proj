#include "TimerUtils.h"

// Define static members in exactly one .cpp file
std::unordered_map<std::string, double> GlobalFunctionTimer::totalTimes;
std::unordered_map<std::string, int> GlobalFunctionTimer::callCounts;