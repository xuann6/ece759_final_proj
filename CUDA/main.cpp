// main_cuda.cpp
#include "rrtCuda.h"               // Standard RRT
#include "rrtStarCuda.h"           // RRT*
#include "rrtBidirectionalCuda.h"  // Bidirectional RRT
#include "rrtInformedCuda.h"       // Informed RRT*
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

// Utility function to print a horizontal separator line
void printSeparator() {
    std::cout << "\n" << std::string(80, '=') << "\n" << std::endl;
}

// Utility function to measure execution time and print statistics
template<typename Func>
void runBenchmark(const std::string& name, Func&& algorithm) {
    std::cout << "Running " << name << "..." << std::endl;
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run algorithm
    auto path = algorithm();
    
    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    // Print results
    if (!path.empty()) {
        std::cout << name << " found a path with " << path.size() << " nodes in "
                  << elapsed.count() << " seconds" << std::endl;
        
        // Calculate path length
        double pathLength = 0.0;
        for (size_t i = 1; i < path.size(); i++) {
            double dx = path[i].x - path[i-1].x;
            double dy = path[i].y - path[i-1].y;
            pathLength += std::sqrt(dx*dx + dy*dy);
        }
        
        std::cout << name << " path length: " << pathLength << std::endl;
        
        // Print first few and last few nodes in the path
        int nodesToPrint = std::min(3, static_cast<int>(path.size()));
        std::cout << "\nPath start: " << std::endl;
        for (int i = 0; i < nodesToPrint; i++) {
            std::cout << "  Node " << i << ": (" << path[i].x << ", " << path[i].y;
            
            // Print cost if available in the Node struct (for RRT* variants)
            if (name.find("RRT*") != std::string::npos && i < path.size()) {
                // Check if the class has a cost field by using dynamic_cast or other means
                // For now, we'll handle this in a bit simpler way:
                if (name.find("Informed") != std::string::npos || name.find("Star") != std::string::npos) {
                    std::cout << "), cost: " << path[i].cost;
                } else {
                    std::cout << ")";
                }
            } else {
                std::cout << ")";
            }
            std::cout << std::endl;
        }
        
        if (path.size() > 2 * nodesToPrint) {
            std::cout << "  ..." << std::endl;
        }
        
        std::cout << "Path end: " << std::endl;
        for (int i = std::max(0, static_cast<int>(path.size()) - nodesToPrint); 
             i < path.size(); i++) {
            std::cout << "  Node " << i << ": (" << path[i].x << ", " << path[i].y;
            
            // Print cost if available in the Node struct (for RRT* variants)
            if (name.find("RRT*") != std::string::npos && i < path.size()) {
                if (name.find("Informed") != std::string::npos || name.find("Star") != std::string::npos) {
                    std::cout << "), cost: " << path[i].cost;
                } else {
                    std::cout << ")";
                }
            } else {
                std::cout << ")";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << name << " failed to find a path in " << elapsed.count() << " seconds" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Configuration
    bool enableVisualization = true;
    int maxIterations = 1000000;       // Default iterations
    int numThreads = 256;           // Default number of CUDA threads
    bool runAll = true;             // Run all algorithms by default
    bool runStandard = false;       // Individual algorithm flags
    bool runStar = false;
    bool runBidirectional = false;
    bool runInformed = false;
    double worldWidth = 1.0;        // Default world width
    double worldHeight = 1.0;       // Default world height
    double stepSize = 0.1;          // Default step size
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-vis") {
            enableVisualization = false;
        } else if (arg == "--iter" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            worldWidth = std::stod(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            worldHeight = std::stod(argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            stepSize = std::stod(argv[++i]);
        } else if (arg == "--standard") {
            runStandard = true;
            runAll = false;
        } else if (arg == "--star") {
            runStar = true;
            runAll = false;
        } else if (arg == "--bi") {
            runBidirectional = true;
            runAll = false;
        } else if (arg == "--informed") {
            runInformed = true;
            runAll = false;
        } else if (arg == "--help") {
            std::cout << "CUDA RRT Implementations\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --no-vis          Disable visualization\n"
                      << "  --iter <num>      Set maximum iterations (default: 5000)\n"
                      << "  --threads <num>   Set number of CUDA threads (default: 256)\n"
                      << "  --width <num>     Set world width (default: 1.0)\n"
                      << "  --height <num>    Set world height (default: 1.0)\n"
                      << "  --step <num>      Set step size (default: 0.1)\n"
                      << "  --standard        Run only standard RRT\n"
                      << "  --star            Run only RRT*\n"
                      << "  --bi              Run only Bidirectional RRT\n"
                      << "  --informed        Run only Informed RRT*\n"
                      << "  --help            Show this help message\n";
            return 0;
        }
    }
    
    // Example usage - create start and goal nodes
    Node start(0.1 * worldWidth, 0.1 * worldHeight);
    Node goal(0.9 * worldWidth, 0.9 * worldHeight);
    
    // Define obstacles
    std::vector<Obstacle> obstacles;
    
    // Create two rectangular obstacles
    double obstacleWidth = worldWidth / 10.0;
    
    // First obstacle at 1/3 of the world width
    double x1 = worldWidth / 3.0 - obstacleWidth / 2.0;
    double height1 = 0.4 * worldHeight;
    obstacles.push_back(Obstacle(x1, 0, obstacleWidth, height1));
    
    // Second obstacle at 2/3 of the world width
    double x2 = 2.0 * worldWidth / 3.0 - obstacleWidth / 2.0;
    double height2 = 0.6 * worldHeight;
    obstacles.push_back(Obstacle(x2, worldHeight - height2, obstacleWidth, height2));
    
    // Print configuration
    std::cout << "CUDA RRT Implementations Benchmark" << std::endl;
    std::cout << "World dimensions: " << worldWidth << " x " << worldHeight << std::endl;
    std::cout << "Start: (" << start.x << ", " << start.y << ")" << std::endl;
    std::cout << "Goal: (" << goal.x << ", " << goal.y << ")" << std::endl;
    std::cout << "Obstacles: " << obstacles.size() << std::endl;
    for (const auto& obstacle : obstacles) {
        std::cout << "  Position: (" << obstacle.x << ", " << obstacle.y << "), "
                 << "Size: " << obstacle.width << " x " << obstacle.height << std::endl;
    }
    std::cout << "Configuration:" << std::endl
              << "  Max iterations: " << maxIterations << std::endl
              << "  CUDA threads: " << numThreads << std::endl
              << "  Step size: " << stepSize << std::endl
              << "  Visualization: " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    
    // Algorithm parameters
    double goalThreshold = 0.1;
    double rewireRadius = 0.2 * worldWidth;  // Radius for RRT* rewiring
    double connectThreshold = 0.15 * worldWidth; // For bidirectional RRT
    
    // Run standard RRT
    if (runAll || runStandard) {
        printSeparator();
        runBenchmark("Standard RRT CUDA", [&]() {
            return buildRRTCuda(
                start, goal, obstacles, stepSize, goalThreshold, maxIterations,
                0.0, worldWidth, 0.0, worldHeight, "rrt_standard_cuda_tree.csv", enableVisualization, numThreads
            );
        });
    }
    
    // Run RRT*
    if (runAll || runStar) {
        printSeparator();
        runBenchmark("RRT* CUDA", [&]() {
            return buildRRTStarCuda(
                start, goal, obstacles, stepSize, goalThreshold, maxIterations, rewireRadius,
                0.0, worldWidth, 0.0, worldHeight, "rrt_star_cuda_tree.csv", enableVisualization, numThreads
            );
        });
    }
    
    // Run Bidirectional RRT
    if (runAll || runBidirectional) {
        printSeparator();
        runBenchmark("Bidirectional RRT CUDA", [&]() {
            return buildRRTBidirectionalCuda(
                start, goal, obstacles, stepSize, connectThreshold, maxIterations,
                0.0, worldWidth, 0.0, worldHeight, "rrt_bi_cuda_tree.csv", enableVisualization, numThreads
            );
        });
    }
    
    // Run Informed RRT*
    if (runAll || runInformed) {
        printSeparator();
        runBenchmark("Informed RRT* CUDA", [&]() {
            return buildRRTInformedCuda(
                start, goal, obstacles, stepSize, goalThreshold, maxIterations, rewireRadius,
                0.0, worldWidth, 0.0, worldHeight, "rrt_informed_cuda_tree.csv", enableVisualization, numThreads, false
            );
        });
    }
    
    printSeparator();
    std::cout << "Benchmarks complete!" << std::endl;
    
    return 0;
}