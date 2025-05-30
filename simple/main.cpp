#include "rrt.h"
#include "rrtStar.h"
#include "rrtInformed.h"
#include "rrtBidirectional.h"
#include <iostream>
#include <chrono>

int main()
{
    // World size configuration
    double worldWidth = 1.0;
    double worldHeight = 1.0;
    double xMin = 0.0, xMax = worldWidth, yMin = 0.0, yMax = worldHeight;
    
    // Configuration
    bool enableVisualization = true; // Set to false to disable visualization and improve performance

    // Example usage - positions as percentages of world size
    double startXPct = 0.1, startYPct = 0.1;
    double goalXPct = 0.9, goalYPct = 0.9;
    
    Node start(startXPct * worldWidth, startYPct * worldHeight);
    Node goal(goalXPct * worldWidth, goalYPct * worldHeight);
    int iterations = 1000000;

    std::cout << "Running RRT from (" << start.x << ", " << start.y << ") to ("
              << goal.x << ", " << goal.y << ")" << std::endl;
    std::cout << "World size: " << worldWidth << " x " << worldHeight << std::endl;

    std::cout << "Visualization is " << (enableVisualization ? "enabled" : "disabled") << std::endl;

    // ===================== RRT Test =====================
    std::cout << "\n======== Standard RRT ========" << std::endl;

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
    std::vector<Node> pathRRT = buildRRT(
        start, goal, 0.1, 0.1, iterations, xMin, xMax, yMin, yMax, "rrt_tree.csv", enableVisualization);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRT = endTime - startTime;

    if (!pathRRT.empty())
    {
        std::cout << "RRT path found with " << pathRRT.size() << " nodes in "
                  << elapsedRRT.count() << " seconds" << std::endl;

        // Calculate path length
        double pathLengthRRT = 0.0;
        for (int i = 1; i < pathRRT.size(); i++)
        {
            pathLengthRRT += distance(pathRRT[i - 1], pathRRT[i]);
        }
        std::cout << "RRT path length: " << pathLengthRRT << std::endl;
    }
    else
    {
        std::cout << "RRT failed to find a path" << std::endl;
    }

    // ===================== RRT* Test =====================
    std::cout << "\n======== RRT* ========" << std::endl;

    auto startTimeRRTStar = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> obstacles;

    // Run RRT*
    std::vector<Node> pathRRTStar = rrt_star::buildRRTStar(
        start, goal, obstacles, 0.1, 0.1, iterations, 0.5, xMin, xMax, yMin, yMax, "rrt_star_tree.csv", enableVisualization, true);

    auto endTimeRRTStar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTStar = endTimeRRTStar - startTimeRRTStar;

    if (!pathRRTStar.empty())
    {
        std::cout << "RRT* path found with " << pathRRTStar.size() << " nodes in "
                  << elapsedRRTStar.count() << " seconds" << std::endl;

        // Calculate path length
        double pathLengthRRTStar = 0.0;
        for (int i = 1; i < pathRRTStar.size(); i++)
        {
            pathLengthRRTStar += distance(pathRRTStar[i - 1], pathRRTStar[i]);
        }
        std::cout << "RRT* path length: " << pathLengthRRTStar << std::endl;
    }
    else
    {
        std::cout << "RRT* failed to find a path" << std::endl;
    }

    // ===================== Informed RRT* Test =====================
    // std::cout << "\n======== Informed RRT* ========" << std::endl;

    // auto startTimeInformedRRTStar = std::chrono::high_resolution_clock::now();

    // // Run Informed RRT*
    // std::vector<Node> pathInformedRRTStar = rrt_informed::buildInformedRRTStar(
    //     start, goal, obstacles, 0.1, 0.1, iterations, 0.5, xMin, xMax, yMin, yMax, "rrt_informed_tree.csv", enableVisualization, true);

    // auto endTimeInformedRRTStar = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsedInformedRRTStar = endTimeInformedRRTStar - startTimeInformedRRTStar;

    // if (!pathInformedRRTStar.empty())
    // {
    //     std::cout << "Informed RRT* path found with " << pathInformedRRTStar.size() << " nodes in "
    //               << elapsedInformedRRTStar.count() << " seconds" << std::endl;

    //     // Calculate path length
    //     double pathLengthInformedRRTStar = 0.0;
    //     for (int i = 1; i < pathInformedRRTStar.size(); i++)
    //     {
    //         pathLengthInformedRRTStar += distance(pathInformedRRTStar[i - 1], pathInformedRRTStar[i]);
    //     }
    //     std::cout << "Informed RRT* path length: " << pathLengthInformedRRTStar << std::endl;
    // }
    // else
    // {
    //     std::cout << "Informed RRT* failed to find a path" << std::endl;
    // }

    // ===================== RRT Bidirectional Test =====================
    std::cout << "\n======== Bidirectional RRT ========" << std::endl;

    // Start timer
    auto bidirectionalRRTStartTime = std::chrono::high_resolution_clock::now();

    // Run RRT-Bi
    // Parameters:
    // start: Starting node of the RRT
    // goal: Goal node of the RRT
    // stepSize: Maximum distance between two nodes in the tree
    // goalThreshold: Distance threshold to consider the goal reached
    // maxIterations: Maximum number of iterations to run the RRT
    // xMin, xMax, yMin, yMax: Bounds of the environment
    // treeFilename: File name to save the tree data for visualization
    // enableVisualization: Flag to enable or disable visualization
    std::vector<Node> pathBidirectionalRRT = bidirectional_rrt::buildBidirectionalRRT(
        start, goal, obstacles, 0.1, 0.1, iterations, xMin, xMax, yMin, yMax, "rrt_bidirectional_tree.csv", enableVisualization);

    // End timer
    auto bidirectionalRRTEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedBidirectionalRRT = bidirectionalRRTEndTime - bidirectionalRRTStartTime;

    if (!pathBidirectionalRRT.empty())
    {
        std::cout << "RRT bidirectional path found with " << pathBidirectionalRRT.size() << " nodes in "
                  << elapsedBidirectionalRRT.count() << " seconds" << std::endl;

        // Calculate path length
        double pathLengthRRTBi = 0.0;
        for (int i = 1; i < pathBidirectionalRRT.size(); i++)
        {
            pathLengthRRTBi += distance(pathBidirectionalRRT[i - 1], pathBidirectionalRRT[i]);
        }
        std::cout << "RRT path length: " << pathLengthRRTBi << std::endl;
    }
    else
    {
        std::cout << "RRT failed to find a path" << std::endl;
    }

    return 0;
}