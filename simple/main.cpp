#include "rrt.h"
#include <iostream>
#include <chrono>

int main() {
    // Configuration
    bool enableVisualization = true;  // Set to false to disable visualization and improve performance
    
    // Example usage
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);
    
    std::cout << "Running RRT from (" << start.x << ", " << start.y << ") to (" 
              << goal.x << ", " << goal.y << ")" << std::endl;
    
    std::cout << "Visualization is " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run RRT
    // Parameters:
    // start: Starting node of the RRT
    // goal: Goal node of the RRT
    // stepSize: Maximum distance between two nodes in the tree
    // goalThreshold: Distance threshold to consider the goal reached
    // maxIterations: Maximum number of iterations to run the RRT
    // xMin, xMax, yMin, yMax: Bounds of the environment
    // treeFilename: File name to save the tree data for visualization
    // enableVisualization: Flag to enable or disable visualization
    std::vector<Node> path = buildRRT(
        start, goal, 0.1, 0.1, 5000, 0.0, 1.0, 0.0, 1.0, "rrt_tree.csv", enableVisualization
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