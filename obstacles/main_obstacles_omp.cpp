#include "rrtOmpWithObstacles.h"
#include "rrtStarWithObstacles.h"
#include "rrtBidirectionalWithObstacles.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>
#include <iomanip>

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
                    double& goalY,
                    bool& runRRTStar,
                    bool& runBidirectionalRRT) {
    // Default values
    numThreads = 4;
    worldWidth = 1.0;
    worldHeight = 1.0;
    enableVisualization = true;
    stepSize = 0.05;
    goalThreshold = 0.05;
    maxIterations = 100000000;
    startX = 0.1 * worldWidth;
    startY = 0.1 * worldHeight;
    goalX = 0.9 * worldWidth;
    goalY = 0.9 * worldHeight;
    runRRTStar = false;
    runBidirectionalRRT = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
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
        } else if (arg == "--run-rrt-star") {
            runRRTStar = true;
        } else if (arg == "--run-bidirectional") {
            runBidirectionalRRT = true;
        } else if (arg == "--run-all") {
            runRRTStar = true;
            runBidirectionalRRT = true;
        } else if (arg == "--help") {
            std::cout << "Multiple RRT With Obstacles Implementations\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --threads <num>        Set number of OpenMP threads (default: 4)\n"
                      << "  --width <num>          Set world width (default: 1.0)\n"
                      << "  --height <num>         Set world height (default: 1.0)\n"
                      << "  --no-vis               Disable visualization\n"
                      << "  --step <num>           Set step size (default: 0.05)\n"
                      << "  --goal-threshold <num> Set goal threshold (default: 0.05)\n"
                      << "  --iter <num>           Set maximum iterations (default: 100000000)\n"
                      << "  --start-x <num>        Set start X coordinate (default: 0.1*width)\n"
                      << "  --start-y <num>        Set start Y coordinate (default: 0.1*height)\n"
                      << "  --goal-x <num>         Set goal X coordinate (default: 0.9*width)\n"
                      << "  --goal-y <num>         Set goal Y coordinate (default: 0.9*height)\n"
                      << "  --run-rrt-star         Run RRT* algorithm in addition to basic RRT\n"
                      << "  --run-bidirectional    Run Bidirectional RRT in addition to basic RRT\n"
                      << "  --run-all              Run all available RRT algorithms\n"
                      << "  --help                 Show this help message\n";
            exit(0);
        }
    }
    
    // If start/goal weren't explicitly provided, set them relative to world size
    if (startX <= 0.0) startX = 0.1 * worldWidth;
    if (startY <= 0.0) startY = 0.1 * worldHeight;
    if (goalX <= 0.0) goalX = 0.9 * worldWidth;
    if (goalY <= 0.0) goalY = 0.9 * worldHeight;
}

// Function to print algorithm performance results
void printResults(const std::string& algorithm, double executionTime, int nodeCount) {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Algorithm: " << algorithm << std::endl;
    std::cout << "Execution time: " << executionTime << " seconds" << std::endl;
    std::cout << "Path nodes: " << nodeCount << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    // Problem parameters
    int numThreads;
    double worldWidth, worldHeight;
    bool enableVisualization;
    double stepSize, goalThreshold;
    int maxIterations;
    double startX, startY, goalX, goalY;
    bool runRRTStar, runBidirectionalRRT;
    
    // Get specifications from command line or use defaults
    getProblemSpecs(argc, argv, numThreads, worldWidth, worldHeight, 
                   enableVisualization, stepSize, goalThreshold, maxIterations,
                   startX, startY, goalX, goalY, runRRTStar, runBidirectionalRRT);
    
    std::cout << "Running RRT implementation with obstacles..." << std::endl;
    std::cout << "World dimensions: " << worldWidth << " x " << worldHeight << std::endl;
    std::cout << "Step size: " << stepSize << ", Goal threshold: " << goalThreshold << std::endl;
    std::cout << "Max iterations: " << maxIterations << std::endl;
    std::cout << "Start: (" << startX << ", " << startY << "), Goal: (" << goalX << ", " << goalY << ")" << std::endl;
    std::cout << "Visualization: " << (enableVisualization ? "enabled" : "disabled") << std::endl;
    std::cout << "Number of threads: " << numThreads << std::endl;
    
    // Generate obstacles - use the same obstacles for all algorithms
    auto obstacles = rrt_omp_obstacles::generateObstacles(worldWidth, worldHeight);
    std::cout << "Generated " << obstacles.size() << " obstacles." << std::endl;
    
    // Store results for all algorithms
    std::vector<std::string> algorithmNames;
    std::vector<double> executionTimes;
    std::vector<int> nodeCounts;
    
    // Variables for timing and measurement
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;
    std::chrono::duration<double> elapsed;
    
    // ========================= Basic RRT with Obstacles (OpenMP) =========================
    // Define start and goal points
    rrt_omp_obstacles::Node startRrtOmp(startX, startY);
    rrt_omp_obstacles::Node goalRrtOmp(goalX, goalY);
    
    // Convert obstacles to the format expected by this algorithm
    std::vector<rrt_omp_obstacles::Obstacle> obstaclesRrtOmp;
    for (const auto& obs : obstacles) {
        obstaclesRrtOmp.push_back(rrt_omp_obstacles::Obstacle(obs.x, obs.y, obs.width, obs.height));
    }
    
    // Record start time
    startTime = std::chrono::high_resolution_clock::now();
    
    // Run the Basic RRT algorithm with obstacles using OpenMP
    auto pathRrtOmp = rrt_omp_obstacles::buildRRTOmpWithObstacles(
        startRrtOmp,
        goalRrtOmp,
        obstaclesRrtOmp,
        stepSize,
        goalThreshold,
        maxIterations,
        0.0, worldWidth,
        0.0, worldHeight,
        "rrt_omp_obstacles_tree.csv",
        enableVisualization,
        numThreads
    );
    
    // Record end time
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    
    // Store results
    algorithmNames.push_back("Basic RRT OpenMP with Obstacles");
    executionTimes.push_back(elapsed.count());
    nodeCounts.push_back(pathRrtOmp.size());
    
    // Print results
    printResults("Basic RRT OpenMP with Obstacles", elapsed.count(), pathRrtOmp.size());
    
    // Run RRT* only if requested
    if (runRRTStar) {
        // ========================= RRT* with Obstacles =========================
        // Define start and goal points
        rrt_star_obstacles::Node startRrtStar(startX, startY);
        rrt_star_obstacles::Node goalRrtStar(goalX, goalY);
        
        // Convert obstacles to the format expected by this algorithm
        std::vector<rrt_star_obstacles::Obstacle> obstaclesRrtStar;
        for (const auto& obs : obstacles) {
            obstaclesRrtStar.push_back(rrt_star_obstacles::Obstacle(obs.x, obs.y, obs.width, obs.height));
        }
        
        // Record start time
        startTime = std::chrono::high_resolution_clock::now();
        
        // Run the RRT* algorithm with obstacles
        auto pathRrtStar = rrt_star_obstacles::buildRRTStarWithObstacles(
            startRrtStar,
            goalRrtStar,
            obstaclesRrtStar,
            stepSize,
            goalThreshold,
            maxIterations,
            0.0, worldWidth,
            0.0, worldHeight,
            stepSize * 5.0, // reasonable neighborhood radius
            "rrt_star_obstacles_tree.csv",
            enableVisualization
        );
        
        // Record end time
        endTime = std::chrono::high_resolution_clock::now();
        elapsed = endTime - startTime;
        
        // Store results
        algorithmNames.push_back("RRT* with Obstacles");
        executionTimes.push_back(elapsed.count());
        nodeCounts.push_back(pathRrtStar.size());
        
        // Print results
        printResults("RRT* with Obstacles", elapsed.count(), pathRrtStar.size());
    }
    
    // Run Bidirectional RRT only if requested
    if (runBidirectionalRRT) {
        // ========================= Bidirectional RRT with Obstacles =========================
        // Define start and goal points
        rrt_bidirectional_obstacles::Node startBiRrt(startX, startY);
        rrt_bidirectional_obstacles::Node goalBiRrt(goalX, goalY);
        
        // Convert obstacles to the format expected by this algorithm
        std::vector<rrt_bidirectional_obstacles::Obstacle> obstaclesBiRrt;
        for (const auto& obs : obstacles) {
            obstaclesBiRrt.push_back(rrt_bidirectional_obstacles::Obstacle(obs.x, obs.y, obs.width, obs.height));
        }
        
        // Record start time
        startTime = std::chrono::high_resolution_clock::now();
        
        // Run the Bidirectional RRT algorithm with obstacles
        auto resultBiRrt = rrt_bidirectional_obstacles::buildBidirectionalRRTWithObstacles(
            startBiRrt,
            goalBiRrt,
            obstaclesBiRrt,
            stepSize,
            goalThreshold,
            maxIterations,
            0.0, worldWidth,
            0.0, worldHeight,
            "rrt_bidirectional_obstacles_start_tree.csv",
            "rrt_bidirectional_obstacles_goal_tree.csv",
            enableVisualization
        );
        
        // Record end time
        endTime = std::chrono::high_resolution_clock::now();
        elapsed = endTime - startTime;
        
        // Store results
        algorithmNames.push_back("Bidirectional RRT with Obstacles");
        executionTimes.push_back(elapsed.count());
        nodeCounts.push_back(resultBiRrt.path.size());
        
        // Print results
        printResults("Bidirectional RRT with Obstacles", elapsed.count(), resultBiRrt.path.size());
        std::cout << "Start tree size: " << resultBiRrt.startTreeSize << ", Goal tree size: " << resultBiRrt.goalTreeSize << std::endl;
    }
    
    // ========================= Summary Table =========================
    // Only show summary table if more than one algorithm was run
    if (algorithmNames.size() > 1) {
        std::cout << "\n\n====== Performance Summary ======" << std::endl;
        std::cout << std::left << std::setw(35) << "Algorithm" 
                  << std::setw(20) << "Execution Time (s)" 
                  << std::setw(15) << "Path Nodes" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        for (size_t i = 0; i < algorithmNames.size(); i++) {
            std::cout << std::left << std::setw(35) << algorithmNames[i]
                      << std::setw(20) << executionTimes[i]
                      << std::setw(15) << nodeCounts[i] << std::endl;
        }
        
        std::cout << std::string(70, '-') << std::endl;
    }
    
    return 0;
} 