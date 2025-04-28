#include "rrtObstaclesOmp.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int numThreads = 4;  // Default number of threads
    
    if (argc > 1) {
        numThreads = std::stoi(argv[1]);
    }
    
    std::cout << "Running RRT with obstacles using " << numThreads << " threads..." << std::endl;
    
    // Define world dimensions and parameters
    double worldWidth = 1.0;
    double worldHeight = 1.0;
    double stepSize = 0.05;
    double goalThreshold = 0.05;
    int maxIterations = 5000;
    
    // Define start and goal points
    rrt_obstacles_omp::Node start(0.1, 0.1);
    rrt_obstacles_omp::Node goal(0.9, 0.9);
    
    // Generate obstacles
    auto obstacles = rrt_obstacles_omp::generateObstacles(worldWidth, worldHeight);
    std::cout << "Generated " << obstacles.size() << " obstacles." << std::endl;
    
    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run the RRT algorithm with obstacles using OpenMP
    auto path = rrt_obstacles_omp::buildRRTWithObstaclesOmp(
        start,
        goal,
        obstacles,
        stepSize,
        goalThreshold,
        maxIterations,
        0.0, worldWidth,
        0.0, worldHeight,
        "rrt_obstacles_omp_tree.csv",
        true,
        numThreads
    );
    
    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    // Print results
    if (!path.empty()) {
        std::cout << "Path found with " << path.size() << " nodes." << std::endl;
        std::cout << "Total execution time: " << elapsed.count() << " seconds." << std::endl;
        
        // Print path points (optional)
        std::cout << "Path points:" << std::endl;
        for (const auto& node : path) {
            std::cout << "(" << node.x << ", " << node.y << ")" << std::endl;
        }
    } else {
        std::cout << "No path found within the maximum number of iterations." << std::endl;
    }
    
    return 0;
} 