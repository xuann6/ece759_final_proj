#include "../simple/rrt.h"
#include "../simple/rrtStar.h"
#include "./rrtOmp.h"
#include "./rrtStarOmp.h"
#include "./rrtBiOmp.h"
#include "./rrtInformedOmp.h"
#include <iostream>
#include <chrono>
#include <iomanip>

int main()
{
    // Configuration
    bool enableVisualization = false; // Set to false to disable visualization and improve performance
    int numThreads = 4;               // Number of threads for OpenMP

    int iterations = 10000;

    // Example usage
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);

    double pathLengthRRT = 0.0;
    double pathLengthRRTOmp = 0.0;
    double pathLengthRRTStar = 0.0;
    double pathLengthRRTStarOmp = 0.0;
    double pathLengthRRTBiOmp = 0.0;
    double pathLengthRRTInformedOmp = 0.0;

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
        start, goal, 0.1, 0.1, iterations, 0.0, 1.0, 0.0, 1.0, "rrt_tree.csv", enableVisualization);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRT = endTime - startTime;

    if (!pathRRT.empty())
    {
        std::cout << "RRT path found with " << pathRRT.size() << " nodes in "
                  << elapsedRRT.count() << " seconds" << std::endl;

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

    // ===================== OpenMP RRT Test =====================
    std::cout << "\n======== Parallel RRT (OpenMP) ========" << std::endl;

    // Start timer
    auto startTimeOmp = std::chrono::high_resolution_clock::now();

    // Run OpenMP RRT
    std::vector<Node> pathRRTOmp = rrt_omp::buildRRTOmp(
        start, goal, 0.1, 0.1, iterations, 0.0, 1.0, 0.0, 1.0, "rrt_omp_tree.csv", enableVisualization, numThreads);

    // End timer
    auto endTimeOmp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTOmp = endTimeOmp - startTimeOmp;

    if (!pathRRTOmp.empty())
    {
        std::cout << "OpenMP RRT path found with " << pathRRTOmp.size() << " nodes in "
                  << elapsedRRTOmp.count() << " seconds" << std::endl;

        for (int i = 1; i < pathRRTOmp.size(); i++)
        {
            pathLengthRRTOmp += distance(pathRRTOmp[i - 1], pathRRTOmp[i]);
        }
        std::cout << "OpenMP RRT path length: " << pathLengthRRTOmp << std::endl;
    }
    else
    {
        std::cout << "OpenMP RRT failed to find a path" << std::endl;
    }

    // ===================== Standard RRT* Test =====================
    std::cout << "\n======== Standard RRT* (Sequential) ========" << std::endl;

    // Start timer
    auto startTimeRRTStar = std::chrono::high_resolution_clock::now();

    // Dummy obstacles (empty vector for simplicity)
    std::vector<std::vector<double>> obstacles;

    // Run RRT*
    std::vector<Node> pathRRTStar = rrt_star::buildRRTStar(
        start, goal, obstacles, 0.1, 0.1, iterations, 0.5, 0.0, 1.0, 0.0, 1.0, "rrt_star_tree.csv", enableVisualization);

    // End timer
    auto endTimeRRTStar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTStar = endTimeRRTStar - startTimeRRTStar;

    if (!pathRRTStar.empty())
    {
        std::cout << "RRT* path found with " << pathRRTStar.size() << " nodes in "
                  << elapsedRRTStar.count() << " seconds" << std::endl;

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

    // ===================== OpenMP RRT* Test =====================
    std::cout << "\n======== Parallel RRT* (OpenMP) ========" << std::endl;

    // Start timer
    auto startTimeRRTStarOmp = std::chrono::high_resolution_clock::now();

    // Run OpenMP RRT*
    std::vector<Node> pathRRTStarOmp = rrt_star_omp::buildRRTStarOmp(
        start, goal, obstacles, 0.1, 0.1, iterations, 0.5, 0.0, 1.0, 0.0, 1.0, "rrt_star_omp_tree.csv", enableVisualization, numThreads);

    // End timer
    auto endTimeRRTStarOmp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTStarOmp = endTimeRRTStarOmp - startTimeRRTStarOmp;

    if (!pathRRTStarOmp.empty())
    {
        std::cout << "OpenMP RRT* path found with " << pathRRTStarOmp.size() << " nodes in "
                  << elapsedRRTStarOmp.count() << " seconds" << std::endl;

        for (int i = 1; i < pathRRTStarOmp.size(); i++)
        {
            pathLengthRRTStarOmp += distance(pathRRTStarOmp[i - 1], pathRRTStarOmp[i]);
        }
        std::cout << "OpenMP RRT* path length: " << pathLengthRRTStarOmp << std::endl;
    }
    else
    {
        std::cout << "OpenMP RRT* failed to find a path" << std::endl;
    }

    // ===================== OpenMP RRT-Bi Test =====================
    std::cout << "\n======== Parallel RRT-Bi (OpenMP) ========" << std::endl;

    // Start timer
    auto startTimeRRTBiOmp = std::chrono::high_resolution_clock::now();

    // Run OpenMP RRT*
    std::vector<Node> pathRRTBiOmp = bidirectional_rrt_omp::buildBidirectionalRRT(
        start, goal, obstacles, 0.1, 0.1, 5000, 0.0, 1.0, 0.0, 1.0, "rrt_bi_omp_tree.csv", enableVisualization, numThreads);

    // End timer
    auto endTimeRRTBiOmp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTBiOmp = endTimeRRTBiOmp - startTimeRRTBiOmp;

    if (!pathRRTBiOmp.empty())
    {
        std::cout << "OpenMP RRT-Bi path found with " << pathRRTBiOmp.size() << " nodes in "
                  << elapsedRRTBiOmp.count() << " seconds" << std::endl;

        for (int i = 1; i < pathRRTBiOmp.size(); i++)
        {
            pathLengthRRTBiOmp += distance(pathRRTBiOmp[i - 1], pathRRTBiOmp[i]);
        }
        std::cout << "OpenMP RRT-Bi path length: " << pathLengthRRTBiOmp << std::endl;
    }
    else
    {
        std::cout << "OpenMP RRT-Bi failed to find a path" << std::endl;
    }

    // ===================== OpenMP RRT-Informed Test =====================
    std::cout << "\n======== Parallel Informed RRT* (OpenMP) ========" << std::endl;

    // Start timer
    auto startTimeRRTInformedOmp = std::chrono::high_resolution_clock::now();

    // Run OpenMP Informed RRT*
    std::vector<Node> pathRRTInformedOmp = rrt_informed_omp::buildInformedRRTStarOmp(
        start, goal, obstacles, 0.1, 0.1, iterations, 0.5, 0.0, 1.0, 0.0, 1.0,
        "rrt_informed_omp_tree.csv", enableVisualization, numThreads, true);

    // End timer
    auto endTimeRRTInformedOmp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRRTInformedOmp = endTimeRRTInformedOmp - startTimeRRTInformedOmp;

    if (!pathRRTInformedOmp.empty())
    {
        std::cout << "OpenMP Informed RRT* path found with " << pathRRTInformedOmp.size() << " nodes in "
                  << elapsedRRTInformedOmp.count() << " seconds" << std::endl;

        for (int i = 1; i < pathRRTInformedOmp.size(); i++)
        {
            pathLengthRRTInformedOmp += distance(pathRRTInformedOmp[i - 1], pathRRTInformedOmp[i]);
        }
        std::cout << "OpenMP Informed RRT* path length: " << pathLengthRRTInformedOmp << std::endl;
    }
    else
    {
        std::cout << "OpenMP Informed RRT* failed to find a path" << std::endl;
    }
    return 0;
}
