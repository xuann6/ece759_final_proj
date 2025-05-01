#include "rrtObstaclesOmp.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>

// Function to parse command line arguments and set problem specifications
void getProblemSpecs(int argc, char* argv[], 
                    int& numThreads, 
                    double& worldWidth, 
                    double& worldHeight, 
                    bool& enableVisualization,
                    double& stepSize, 
                    double& goalThreshold, 
                    int& maxIterations,
                    double& startX, 
                    double& startY, 
                    double& goalX, 
                    double& goalY) {
    // Default values
    numThreads = 4;
    worldWidth = 1.0;
    worldHeight = 1.0;
    enableVisualization = false;
    stepSize = 0.05;
    goalThreshold = 0.05;
    maxIterations = 100000000;
    startX = 0.1 * worldWidth;
    startY = 0.1 * worldHeight;
    goalX = 0.9 * worldWidth;
    goalY = 0.9 * worldHeight;
    
    // Parse command line arguments if provided
    if (argc > 1) numThreads = std::stoi(argv[1]);
    if (argc > 2) worldWidth = std::stod(argv[2]);
    if (argc > 3) worldHeight = std::stod(argv[3]);
    if (argc > 4) enableVisualization = (std::stoi(argv[4]) != 0);
    if (argc > 5) stepSize = std::stod(argv[5]);
    if (argc > 6) goalThreshold = std::stod(argv[6]);
    if (argc > 7) maxIterations = std::stoi(argv[7]);
    if (argc > 8) startX = std::stod(argv[8]);
    if (argc > 9) startY = std::stod(argv[9]);
    if (argc > 10) goalX = std::stod(argv[10]);
    if (argc > 11) goalY = std::stod(argv[11]);
    
    // If start/goal weren't explicitly provided, set them relative to world size
    if (argc <= 8) startX = 0.1 * worldWidth;
    if (argc <= 9) startY = 0.1 * worldHeight;
    if (argc <= 10) goalX = 0.9 * worldWidth;
    if (argc <= 11) goalY = 0.9 * worldHeight;
}

int main(int argc, char* argv[]) {
    // Problem parameters
    int numThreads;
    double worldWidth, worldHeight;
    bool enableVisualization;
    double stepSize, goalThreshold;
    int maxIterations;
    double startX, startY, goalX, goalY;
    
    // Get specifications from command line or use defaults
    getProblemSpecs(argc, argv, numThreads, worldWidth, worldHeight, 
                   enableVisualization, stepSize, goalThreshold, maxIterations,
                   startX, startY, goalX, goalY);
    
    std::cout << "Running RRT with obstacles using " << numThreads << " threads..." << std::endl;
    std::cout << "World dimensions: " << worldWidth << " x " << worldHeight << std::endl;
    std::cout << "Step size: " << stepSize << ", Goal threshold: " << goalThreshold << std::endl;
    std::cout << "Max iterations: " << maxIterations << std::endl;
    std::cout << "Start: (" << startX << ", " << startY << "), Goal: (" << goalX << ", " << goalY << ")" << std::endl;
    std::cout << "Visualization: " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    
    // Define start and goal points
    rrt_obstacles_omp::Node start(startX, startY);
    rrt_obstacles_omp::Node goal(goalX, goalY);
    
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
        enableVisualization,
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
        // std::cout << "Path points:" << std::endl;
        // for (const auto& node : path) {
        //     std::cout << "(" << node.x << ", " << node.y << ")" << std::endl;
        // }
    } else {
        std::cout << "No path found within the maximum number of iterations." << std::endl;
    }
    
    return 0;
} 