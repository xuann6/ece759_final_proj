#include "rrtStarOmp.h"
#include "../simple/rrtStar.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

namespace rrt_star_omp {

    // Find nodes within a certain radius in parallel
    std::vector<int> findNearNodesParallel(const std::vector<Node>& nodes, const Node& newNode, double radius) {
        std::vector<int> nearIndices;
        
        #pragma omp parallel
        {
            std::vector<int> local_nearIndices;
            
            #pragma omp for nowait
            for (int i = 0; i < nodes.size(); i++) {
                if (distance(nodes[i], newNode) <= radius) {
                    local_nearIndices.push_back(i);
                }
            }
            
            #pragma omp critical
            {
                nearIndices.insert(nearIndices.end(), local_nearIndices.begin(), local_nearIndices.end());
            }
        }
        
        return nearIndices;
    }

    // Choose best parent for a new node (RRT* specific) in parallel
    int chooseBestParentParallel(const std::vector<Node>& nodes, const Node& newNode, 
                               const std::vector<int>& nearIndices, 
                               const std::vector<std::vector<double>>& obstacles) {
        int bestParentIndex = -1;
        double bestCost = std::numeric_limits<double>::infinity();
        
        #pragma omp parallel
        {
            int local_bestParentIndex = -1;
            double local_bestCost = std::numeric_limits<double>::infinity();
            
            #pragma omp for nowait
            for (int i = 0; i < nearIndices.size(); i++) {
                int idx = nearIndices[i];
                
                // Check if the path is collision-free
                if (rrt_star::isPathClear(nodes[idx], newNode, obstacles)) {
                    // Calculate cost through this potential parent
                    double costThroughParent = nodes[idx].cost + distance(nodes[idx], newNode);
                    
                    // Update best parent if this is better
                    if (costThroughParent < local_bestCost) {
                        local_bestCost = costThroughParent;
                        local_bestParentIndex = idx;
                    }
                }
            }
            
            // Critical section to update the global best
            #pragma omp critical
            {
                if (local_bestCost < bestCost) {
                    bestCost = local_bestCost;
                    bestParentIndex = local_bestParentIndex;
                }
            }
        }
        
        return bestParentIndex;
    }

    // Structure to represent rewiring operations
    struct RewireOperation {
        int nodeIdx;
        double newCost;
    };

    // Rewire the tree to optimize paths (RRT* specific) with parallel execution
    void rewireTreeParallel(std::vector<Node>& nodes, int newNodeIdx, 
                          const std::vector<int>& nearIndices,
                          const std::vector<std::vector<double>>& obstacles) {
        const Node& newNode = nodes[newNodeIdx];
        std::vector<RewireOperation> rewireOps;
        
        #pragma omp parallel
        {
            std::vector<RewireOperation> local_rewireOps;
            
            #pragma omp for nowait
            for (int i = 0; i < nearIndices.size(); i++) {
                int nearIdx = nearIndices[i];
                
                // Skip the parent of the new node
                if (nearIdx == nodes[newNodeIdx].parent) {
                    continue;
                }
                
                // Check if the path is collision-free
                if (rrt_star::isPathClear(newNode, nodes[nearIdx], obstacles)) {
                    // Calculate cost through the new node
                    double costThroughNew = newNode.cost + distance(newNode, nodes[nearIdx]);
                    
                    // Rewire if the cost is lower
                    if (costThroughNew < nodes[nearIdx].cost) {
                        local_rewireOps.push_back({nearIdx, costThroughNew});
                    }
                }
            }
            
            #pragma omp critical
            {
                rewireOps.insert(rewireOps.end(), local_rewireOps.begin(), local_rewireOps.end());
            }
        }
        
        // Apply all rewiring operations (must be done sequentially to avoid race conditions)
        for (const auto& op : rewireOps) {
            // Update parent and cost
            nodes[op.nodeIdx].parent = newNodeIdx;
            nodes[op.nodeIdx].cost = op.newCost;
            
            // Recursively update costs of children
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i].parent == op.nodeIdx) {
                    // Update cost
                    double oldCost = nodes[i].cost;
                    nodes[i].cost = nodes[op.nodeIdx].cost + distance(nodes[op.nodeIdx], nodes[i]);
                    
                    // If cost changed, check descendants of this node too
                    if (std::abs(nodes[i].cost - oldCost) > 1e-6) {
                        std::vector<int> childIndices = {i};
                        rrt_star::rewireTree(nodes, i, childIndices, obstacles);
                    }
                }
            }
        }
    }
    
    // Main Parallel RRT* algorithm
    std::vector<Node> buildRRTStarOmp(
        const Node& start,
        const Node& goal,
        const std::vector<std::vector<double>>& obstacles,
        double stepSize,
        double goalThreshold,
        int maxIterations,
        double rewireRadius,
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
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime = 
            std::chrono::high_resolution_clock::now();
        
        // Random number generation setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xDist(xMin, xMax);
        std::uniform_real_distribution<> yDist(yMin, yMax);
        
        // Initialize tree with start node (cost = 0)
        std::vector<Node> nodes;
        nodes.push_back(Node(start.x, start.y, -1, 0.0, 0.0)); // Start node at time 0, cost 0
        
        // Best solution found so far
        double bestCost = std::numeric_limits<double>::infinity();
        int goalNodeIndex = -1;
        
        // Main loop
        for (int i = 0; i < maxIterations; i++) {
            // Get current time for this iteration
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;
            double timeSeconds = elapsed.count();
            
            // Generate random node (with small probability, sample the goal)
            Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.05) ? 
                             goal : Node(xDist(gen), yDist(gen));
            
            // Find nearest node using parallel implementation
            int nearestIndex = rrt_omp::findNearestParallel(nodes, randomNode);
            
            // Create new node by steering
            Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
            newNode.time = timeSeconds;
            
            // Check if path to new node is collision-free
            if (rrt_star::isPathClear(nodes[nearestIndex], newNode, obstacles)) {
                // Find nodes within the rewiring radius (parallel)
                std::vector<int> nearIndices = findNearNodesParallel(nodes, newNode, rewireRadius);
                
                // Choose best parent (parallel)
                int bestParentIndex = chooseBestParentParallel(nodes, newNode, nearIndices, obstacles);
                
                if (bestParentIndex != -1) {
                    // Set parent and cost for the new node
                    newNode.parent = bestParentIndex;
                    newNode.cost = nodes[bestParentIndex].cost + distance(nodes[bestParentIndex], newNode);
                    
                    // Add new node to tree
                    nodes.push_back(newNode);
                    int newNodeIndex = nodes.size() - 1;
                    
                    // Rewire the tree (parallel)
                    rewireTreeParallel(nodes, newNodeIndex, nearIndices, obstacles);
                    
                    // Check if we can reach the goal from this new node
                    double distToGoal = distance(newNode, goal);
                    if (distToGoal <= goalThreshold) {
                        // Check if path to goal is collision-free
                        if (rrt_star::isPathClear(newNode, goal, obstacles)) {
                            // Calculate total cost to goal
                            double totalCost = newNode.cost + distToGoal;
                            
                            // If this path is better than previous solutions
                            if (totalCost < bestCost) {
                                bestCost = totalCost;
                                
                                // Create goal node
                                Node goalNode = goal;
                                goalNode.parent = newNodeIndex;
                                goalNode.cost = totalCost;
                                
                                // Set time for goal node
                                auto goalTime = std::chrono::high_resolution_clock::now();
                                std::chrono::duration<double> goalElapsed = goalTime - startTime;
                                goalNode.time = goalElapsed.count();
                                
                                // Add or update goal node
                                if (goalNodeIndex == -1) {
                                    nodes.push_back(goalNode);
                                    goalNodeIndex = nodes.size() - 1;
                                } else {
                                    // Replace existing goal node with better path
                                    nodes[goalNodeIndex] = goalNode;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Save the tree data if visualization is enabled
        if (enableVisualization) {
            saveTreeToFile(nodes, treeFilename);
        }
        
        // If goal was reached, extract and return the path
        if (goalNodeIndex != -1) {
            return extractPath(nodes, goalNodeIndex);
        } else {
            // If goal not reached, return empty path
            std::cout << "Goal not reached within max iterations." << std::endl;
            return std::vector<Node>();
        }
    }

} // end of namespace