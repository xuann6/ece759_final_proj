// TimerUtils.h
#ifndef TIMER_UTILS_H
#define TIMER_UTILS_H

#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <set>

// Global timer class that will be shared across namespaces
class GlobalFunctionTimer {
private:
    // Declare static members (without defining them)
    static std::unordered_map<std::string, double> totalTimes;
    static std::unordered_map<std::string, int> callCounts;
    static std::set<std::string> excludedFunctions;
    
    std::string functionName;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

public:
    GlobalFunctionTimer(const std::string& name) : functionName(name) {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    ~GlobalFunctionTimer() {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        totalTimes[functionName] += elapsed.count();
        callCounts[functionName]++;
    }
    
    static void printStatistics() {
        std::cout << "\n--- Function Timing Statistics ---\n";
        double totalTime = 0.0;
        
        // Initialize excluded functions if not already done
        if (excludedFunctions.empty()) {
            excludedFunctions.insert("saveTreeToFile");
            excludedFunctions.insert("saveTreesToFile");
        }
        
        // Calculate the total time spent in all non-excluded functions
        for (const auto& entry : totalTimes) {
            // Skip excluded functions for the total calculation
            if (excludedFunctions.find(entry.first) != excludedFunctions.end()) {
                continue;
            }
            totalTime += entry.second;
        }
        
        if (totalTime == 0.0) {
            std::cout << "No timing data available\n";
            return;
        }
        
        // Print statistics for each function
        for (const auto& entry : totalTimes) {
            const std::string& funcName = entry.first;
            
            // Skip excluded functions completely if desired
            // Uncomment the next 3 lines to hide excluded functions entirely
            /*if (excludedFunctions.find(funcName) != excludedFunctions.end()) {
                continue;
            }*/
            
            double funcTotalTime = entry.second;
            int count = callCounts[funcName];
            
            std::cout << "Function: " << funcName << "\n";
            std::cout << "  Total calls: " << count << "\n";
            std::cout << "  Total time: " << funcTotalTime << " seconds\n";
            std::cout << "  Average time per call: " << (funcTotalTime / count) << " seconds\n";
            
            // Only calculate percentage for non-excluded functions
            if (excludedFunctions.find(funcName) == excludedFunctions.end()) {
                std::cout << "  Percentage of total: " << (funcTotalTime / totalTime * 100) << "%\n";
            } else {
                std::cout << "  Percentage of total: [excluded from calculation]\n";
            }
            std::cout << "\n";
        }
    }
    
    static void reset() {
        totalTimes.clear();
        callCounts.clear();
    }
};

#endif // TIMER_UTILS_H