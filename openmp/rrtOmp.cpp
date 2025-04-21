#include "rrtOmp.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

namespace rrt_omp {

    // Find nearest node in parallel
    int findNearestParallel(const std::vector<Node>& nodes, const Node& point) {
        int nearest = 0;
        double minDist = distance(nodes[0], point);
        
        #pragma omp parallel
        {
            int local_nearest = 0;
            double local_minDist = minDist;
            
            #pragma omp for nowait
            for (int i = 1; i < nodes.size(); i++) {
                double dist = distance(nodes[i], point);
                if (dist < local_minDist) {
                    local_minDist = dist;
                    local_nearest = i;
                }
            }
            
            #pragma omp critical
            {
                if (local_minDist < minDist) {
                    minDist = local_minDist;
                    nearest = local_nearest;
                }
            }
        }
        
        return nearest;
    }
    
    // Main RRT algorithm with OpenMP parallelization
    std::vector<Node> buildRRTOmp(
        const Node& start,
        const Node& goal,
        double stepSize,
        double goalThreshold,
        int maxIterations,
        double xMin,
        double xMax,
        double yMin,
        double yMax,
        const std::string& treeFilename,
        bool enableVisualization,
        int numThreads
    ) {
        // Set the number of OpenMP threads
        omp_set_num_threads(numThreads);
        
        // Start timing for all runs
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
        
        // Random number generation setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xDist(xMin, xMax);
        std::uniform_real_distribution<> yDist(yMin, yMax);
        
        // Initialize tree with start node
        std::vector<Node> nodes;
        nodes.push_back(Node(start.x, start.y, -1, 0.0)); // Start node at time 0
        
        // Main loop
        for (int i = 0; i < maxIterations; i++) {
            // Get current time for this iteration
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;
            double timeSeconds = elapsed.count();
            
            // Generate random node (with small probability, sample the goal)
            Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.05) ? 
                              goal : Node(xDist(gen), yDist(gen));
            
            // Find nearest node (parallel version)
            int nearestIndex = findNearestParallel(nodes, randomNode);
            
            // Create new node by steering
            Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
            newNode.parent = nearestIndex;
            newNode.time = timeSeconds;
            
            // Add new node to tree
            nodes.push_back(newNode);
            int newNodeIndex = nodes.size() - 1;
            
            // Check if goal reached
            if (distance(newNode, goal) <= goalThreshold) {
                // Add goal node to tree
                Node goalNode = goal;
                goalNode.parent = newNodeIndex;
                
                // Set time for goal node
                auto goalTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> goalElapsed = goalTime - startTime;
                goalNode.time = goalElapsed.count();
                
                nodes.push_back(goalNode);
                
                // Save the tree data if visualization is enabled
                if (enableVisualization) {
                    saveTreeToFile(nodes, treeFilename);
                }
                
                // Extract and return path
                return extractPath(nodes, nodes.size() - 1);
            }
        }
        
        // If goal not reached, save tree anyway (if visualization is enabled)
        if (enableVisualization) {
            saveTreeToFile(nodes, treeFilename);
        }
        
        // If goal not reached, return empty path
        std::cout << "Goal not reached within max iterations." << std::endl;
        return std::vector<Node>();
    }
    
} // end of namespace