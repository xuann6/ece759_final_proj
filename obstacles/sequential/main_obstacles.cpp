#include "rrt_obstacles.h"
#include <iostream>
#include <chrono>

int main() {
    // Configuration
    bool enableVisualization = true;  // Set to false to disable visualization and improve performance
    
    // Example usage
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);
    
    // World bounds
    double xMin = 0.0;
    double xMax = 1.0;
    double yMin = 0.0;
    double yMax = 1.0;
    
    // Generate obstacles
    std::vector<Obstacle> obstacles = generateObstacles(xMax - xMin, yMax - yMin);
    
    std::cout << "Running RRT with obstacles from (" << start.x << ", " << start.y << ") to (" 
              << goal.x << ", " << goal.y << ")" << std::endl;
              
    std::cout << "Obstacles:" << std::endl;
    for (const auto& obstacle : obstacles) {
        std::cout << "  Position: (" << obstacle.x << ", " << obstacle.y << "), "
                  << "Size: " << obstacle.width << " x " << obstacle.height << std::endl;
    }
    
    std::cout << "Visualization is " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run RRT with obstacles
    // Parameters:
    // start: Starting node of the RRT
    // goal: Goal node of the RRT
    // obstacles: List of obstacles to avoid
    // stepSize: Maximum distance between two nodes in the tree
    // goalThreshold: Distance threshold to consider the goal reached
    // maxIterations: Maximum number of iterations to run the RRT
    // xMin, xMax, yMin, yMax: Bounds of the environment
    // treeFilename: File name to save the tree data for visualization
    // enableVisualization: Flag to enable or disable visualization
    std::vector<Node> path = buildRRTWithObstacles(
        start, goal, obstacles, 0.1, 0.1, 10000, xMin, xMax, yMin, yMax, 
        "rrt_obstacles_tree.csv", enableVisualization
    );
    
    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    if (!path.empty()) {
        std::cout << "Path found with " << path.size() << " nodes in " 
                  << elapsed.count() << " seconds:" << std::endl;
        
        for (const auto& node : path) {
            std::cout << "(" << node.x << ", " << node.y << ")" << std::endl;
        }
    }
    
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
        
    return 0;
} 