#include "rrtStar.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace rrt_star {
    // Find nodes within a certain radius
    std::vector<int> findNearNodes(const std::vector<Node>& nodes, const Node& newNode, double radius) {
        std::vector<int> nearIndices;
        
        for (int i = 0; i < nodes.size(); i++) {
            if (distance(nodes[i], newNode) <= radius) {
                nearIndices.push_back(i);
            }
        }
        
        return nearIndices;
    }


    // Check if the path between two nodes is collision-free
    bool isPathClear(const Node& from, const Node& to, const std::vector<std::vector<double>>& obstacles) {
        // For each obstacle (represented as [x, y, radius])
        
        if (obstacles.empty()) {
            return true;  // No obstacles, for the easiest case
        }
        
        for (const auto& obstacle : obstacles) {
            double ox = obstacle[0];
            double oy = obstacle[1];
            double radius = obstacle[2];
            
            // Vector from 'from' to 'to'
            double dx = to.x - from.x;
            double dy = to.y - from.y;
            double length = std::sqrt(dx * dx + dy * dy);
            
            // Normalize direction vector
            if (length > 0) {
                dx /= length;
                dy /= length;
            }
            
            // Vector from 'from' to obstacle center
            double ox_f = ox - from.x;
            double oy_f = oy - from.y;
            
            // Calculate projection of obstacle center onto the line
            double t = ox_f * dx + oy_f * dy;
            
            // Find closest point on the line to obstacle center
            double closestX, closestY;
            
            if (t < 0) {
                // Closest point is 'from'
                closestX = from.x;
                closestY = from.y;
            } else if (t > length) {
                // Closest point is 'to'
                closestX = to.x;
                closestY = to.y;
            } else {
                // Closest point is on the line segment
                closestX = from.x + t * dx;
                closestY = from.y + t * dy;
            }
            
            // Check if the closest point is within the obstacle
            double distToObstacle = std::sqrt(
                (closestX - ox) * (closestX - ox) + 
                (closestY - oy) * (closestY - oy)
            );
            
            if (distToObstacle <= radius) {
                return false; // Collision detected
            }
        }
        
        return true; // No collision
    }

    // Choose best parent for a new node (RRT* specific)
    int chooseBestParent(const std::vector<Node>& nodes, const Node& newNode, 
                        const std::vector<int>& nearIndices, 
                        const std::vector<std::vector<double>>& obstacles) {
        int bestParentIndex = -1;
        double bestCost = std::numeric_limits<double>::infinity();
        
        for (int idx : nearIndices) {
            // Check if the path is collision-free
            if (isPathClear(nodes[idx], newNode, obstacles)) {
                // Calculate cost through this potential parent
                double costThroughParent = nodes[idx].cost + distance(nodes[idx], newNode);
                
                // Update best parent if this is better
                if (costThroughParent < bestCost) {
                    bestCost = costThroughParent;
                    bestParentIndex = idx;
                }
            }
        }
        
        return bestParentIndex;
    }

    // Rewire the tree to optimize paths (RRT* specific)
    void rewireTree(std::vector<Node>& nodes, int newNodeIdx, 
                const std::vector<int>& nearIndices,
                const std::vector<std::vector<double>>& obstacles) {
        const Node& newNode = nodes[newNodeIdx];
        
        for (int nearIdx : nearIndices) {
            // Skip the parent of the new node
            if (nearIdx == nodes[newNodeIdx].parent) {
                continue;
            }
            
            // Check if the path is collision-free
            if (isPathClear(newNode, nodes[nearIdx], obstacles)) {
                // Calculate cost through the new node
                double costThroughNew = newNode.cost + distance(newNode, nodes[nearIdx]);
                
                // Rewire if the cost is lower
                if (costThroughNew < nodes[nearIdx].cost) {
                    nodes[nearIdx].parent = newNodeIdx;
                    nodes[nearIdx].cost = costThroughNew;
                    
                    // Propagate cost changes to descendants (recursive approach)
                    // In a production system, you might use a more efficient approach
                    for (int i = 0; i < nodes.size(); i++) {
                        if (nodes[i].parent == nearIdx) {
                            // Update cost
                            double oldCost = nodes[i].cost;
                            nodes[i].cost = nodes[nearIdx].cost + distance(nodes[nearIdx], nodes[i]);
                            
                            // If cost changed, check descendants of this node too
                            if (nodes[i].cost != oldCost) {
                                std::vector<int> childIndices = {i};
                                rewireTree(nodes, i, childIndices, obstacles);
                            }
                        }
                    }
                }
            }
        }
    }


// Main RRT* algorithm
std::vector<Node> buildRRTStar(
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
    bool enableVisualization
) {
    // Start timing for all runs
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
    
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
        
        // Find nearest node
        int nearestIndex = findNearest(nodes, randomNode);
        
        // Create new node by steering
        Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
        newNode.time = timeSeconds;
        
        // Check if path to new node is collision-free
        if (isPathClear(nodes[nearestIndex], newNode, obstacles)) {
            // Find nodes within the rewiring radius
            std::vector<int> nearIndices = findNearNodes(nodes, newNode, rewireRadius);
            
            // Choose best parent
            int bestParentIndex = chooseBestParent(nodes, newNode, nearIndices, obstacles);
            
            if (bestParentIndex != -1) {
                // Set parent and cost for the new node
                newNode.parent = bestParentIndex;
                newNode.cost = nodes[bestParentIndex].cost + distance(nodes[bestParentIndex], newNode);
                
                // Add new node to tree
                nodes.push_back(newNode);
                int newNodeIndex = nodes.size() - 1;
                
                // Rewire the tree
                rewireTree(nodes, newNodeIndex, nearIndices, obstacles);
                
                // Check if we can reach the goal from this new node
                double distToGoal = distance(newNode, goal);
                if (distToGoal <= goalThreshold) {
                    // Check if path to goal is collision-free
                    if (isPathClear(newNode, goal, obstacles)) {
                        
                        // Todo: this is one of the option, need further discussion here
                        // double totalCost = newNode.cost + distToGoal;
                        
                        // // If this path is better than previous solutions
                        // if (totalCost < bestCost) {
                        //     bestCost = totalCost;
                            
                        //     // Create goal node
                        //     Node goalNode = goal;
                        //     goalNode.parent = newNodeIndex;
                        //     goalNode.cost = totalCost;
                            
                        //     // Set time for goal node
                        //     auto goalTime = std::chrono::high_resolution_clock::now();
                        //     std::chrono::duration<double> goalElapsed = goalTime - startTime;
                        //     goalNode.time = goalElapsed.count();
                            
                        //     // Add or update goal node
                        //     if (goalNodeIndex == -1) {
                        //         nodes.push_back(goalNode);
                        //         goalNodeIndex = nodes.size() - 1;
                        //     } else {
                        //         // Replace existing goal node with better path
                        //         nodes[goalNodeIndex] = goalNode;
                        //     }
                        // }

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

} // end of using namespace