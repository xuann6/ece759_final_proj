#include "../simple/rrt.h"
#include "../simple/rrtStar.h"
#include "./rrtOmp.h"
#include <iostream>
#include <chrono>
#include <iomanip>

int main() {
    // Configuration
    bool enableVisualization = true;  // Set to false to disable visualization and improve performance
    int numThreads = 4;               // Number of threads for OpenMP
    
    // Example usage
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);

    double pathLengthRRT = 0.0;
    double pathLengthRRTOmp = 0.0;
    double pathLengthRRTStar = 0.0;
    
    std::cout << "Running RRT from (" << start.x << ", " << start.y << ") to (" 
              << goal.x << ", " << goal.y << ")" << std::endl;
    
    std::cout << "Visualization is " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    std::cout << "Using " << numThreads << " threads for parallel execution" << std::endl;
    
    // ===================== Standard RRT Test =====================
    std::cout << "\n======== Standard RRT (Sequential) ========" << std::endl;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run standard RRT
    std::vector<Node> pathRRT = buildRRT(
        start, goal, 0.1, 0.1, 5000, 0.0, 1.0, 0.0, 1.0, "rrt_tree.csv", enableVisualization
    );
    
    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRT = endTime - startTime;
    
    if (!pathRRT.empty()) {
        std::cout << "RRT path found with " << pathRRT.size() << " nodes in "
                << elapsedRRT.count() << " seconds" << std::endl;
                
        for (int i = 1; i < pathRRT.size(); i++) {
            pathLengthRRT += distance(pathRRT[i-1], pathRRT[i]);
        }
        std::cout << "RRT path length: " << pathLengthRRT << std::endl;
    } else {
        std::cout << "RRT failed to find a path" << std::endl;
    }
    
    // ===================== OpenMP RRT Test =====================
    std::cout << "\n======== Parallel RRT (OpenMP) ========" << std::endl;
    
    // Start timer
    auto startTimeOmp = std::chrono::high_resolution_clock::now();
    
    // Run OpenMP RRT
    std::vector<Node> pathRRTOmp = rrt_omp::buildRRTOmp(
        start, goal, 0.1, 0.1, 5000, 0.0, 1.0, 0.0, 1.0, "rrt_omp_tree.csv", enableVisualization, numThreads
    );
    
    // End timer
    auto endTimeOmp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTOmp = endTimeOmp - startTimeOmp;
    

    if (!pathRRTOmp.empty()) {
        std::cout << "OpenMP RRT path found with " << pathRRTOmp.size() << " nodes in "
                << elapsedRRTOmp.count() << " seconds" << std::endl;
        
        for (int i = 1; i < pathRRTOmp.size(); i++) {
            pathLengthRRTOmp += distance(pathRRTOmp[i-1], pathRRTOmp[i]);
        }
        std::cout << "OpenMP RRT path length: " << pathLengthRRTOmp << std::endl;
    } else {
        std::cout << "OpenMP RRT failed to find a path" << std::endl;
    }
    
    return 0;
}