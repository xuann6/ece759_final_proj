#include "rrt_obstacles.h"
#include <iostream>
#include <chrono>
#include <string>

// Function to parse command line arguments and set problem specifications
void getProblemSpecs(int argc, char* argv[], 
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
    worldWidth = 100.0;
    worldHeight = 100.0;
    enableVisualization = true;
    stepSize = 0.1;
    goalThreshold = 0.1;
    maxIterations = 1000000;
    startX = 0.1 * worldWidth;
    startY = 0.1 * worldHeight;
    goalX = 0.9 * worldWidth;
    goalY = 0.9 * worldHeight;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            worldWidth = std::stod(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            worldHeight = std::stod(argv[++i]);
        } else if (arg == "--no-vis") {
            enableVisualization = false;
        } else if (arg == "--step" && i + 1 < argc) {
            stepSize = std::stod(argv[++i]);
        } else if (arg == "--goal-threshold" && i + 1 < argc) {
            goalThreshold = std::stod(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "--start-x" && i + 1 < argc) {
            startX = std::stod(argv[++i]);
        } else if (arg == "--start-y" && i + 1 < argc) {
            startY = std::stod(argv[++i]);
        } else if (arg == "--goal-x" && i + 1 < argc) {
            goalX = std::stod(argv[++i]);
        } else if (arg == "--goal-y" && i + 1 < argc) {
            goalY = std::stod(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "RRT With Obstacles - Sequential Implementation\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --width <num>          Set world width (default: 100.0)\n"
                      << "  --height <num>         Set world height (default: 100.0)\n"
                      << "  --no-vis               Disable visualization\n"
                      << "  --step <num>           Set step size (default: 0.1)\n"
                      << "  --goal-threshold <num> Set goal threshold (default: 0.1)\n"
                      << "  --iter <num>           Set maximum iterations (default: 1000000)\n"
                      << "  --start-x <num>        Set start X coordinate (default: 0.1*width)\n"
                      << "  --start-y <num>        Set start Y coordinate (default: 0.1*height)\n"
                      << "  --goal-x <num>         Set goal X coordinate (default: 0.9*width)\n"
                      << "  --goal-y <num>         Set goal Y coordinate (default: 0.9*height)\n"
                      << "  --help                 Show this help message\n";
            exit(0);
        }
    }
    
        startX = 0.1 * worldWidth;
        startY = 0.1 * worldHeight;
        goalX = 0.9 * worldWidth;
        goalY = 0.9 * worldHeight;
}

int main(int argc, char* argv[]) {
    // Problem parameters
    double worldWidth, worldHeight;
    bool enableVisualization;
    double stepSize, goalThreshold;
    int maxIterations;
    double startX, startY, goalX, goalY;
    
    // Get specifications from command line or use defaults
    getProblemSpecs(argc, argv, worldWidth, worldHeight, 
                   enableVisualization, stepSize, goalThreshold, maxIterations,
                   startX, startY, goalX, goalY);
    
    // World bounds
    double xMin = 0.0;
    double xMax = worldWidth;
    double yMin = 0.0;
    double yMax = worldHeight;
    
    // Start and goal positions
    Node start(startX, startY);
    Node goal(goalX, goalY);
    
    // Generate obstacles
    std::vector<Obstacle> obstacles = generateObstacles(worldWidth, worldHeight);
    
    std::cout << "Running RRT with obstacles from (" << start.x << ", " << start.y << ") to (" 
              << goal.x << ", " << goal.y << ")" << std::endl;
    std::cout << "World dimensions: " << worldWidth << " x " << worldHeight << std::endl;
    std::cout << "Step size: " << stepSize << ", Goal threshold: " << goalThreshold << std::endl;
    std::cout << "Max iterations: " << maxIterations << std::endl;
    std::cout << "Visualization is " << (enableVisualization ? "enabled" : "disabled") << std::endl;
              
    std::cout << "Obstacles:" << std::endl;
    for (const auto& obstacle : obstacles) {
        std::cout << "  Position: (" << obstacle.x << ", " << obstacle.y << "), "
                  << "Size: " << obstacle.width << " x " << obstacle.height << std::endl;
    }
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run RRT with obstacles
    std::vector<Node> path = buildRRTWithObstacles(
        start, goal, obstacles, stepSize, goalThreshold, maxIterations, 
        xMin, xMax, yMin, yMax, "rrt_obstacles_tree.csv", enableVisualization
    );
    
    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    if (!path.empty()) {
        std::cout << "Path found with " << path.size() << " nodes in " 
                  << elapsed.count() << " seconds:" << std::endl;
        }
    else {
        std::cout << "No path found after " << elapsed.count() << " seconds" << std::endl;
    }
    
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
        
    return 0;
} 