#include "rrtStar.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <queue>

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
        //std::cout << "Entered isPathClear." << std::endl;

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
                //std::cout << "Left isPathClear." << std::endl;
                return false; // Collision detected
            }
        }

        //std::cout << "Left isPathClear." << std::endl;
        return true; // No collision
    }

    // Choose best parent for a new node (RRT* specific)
    int chooseBestParent(const std::vector<Node>& nodes, const Node& newNode, 
                        const std::vector<int>& nearIndices, 
                        const std::vector<std::vector<double>>& obstacles) {
        
        //std::cout << "Entered chooseBestParent." << std::endl;
        
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
        
        //std::cout << "Left chooseBestParent." << std::endl;
        
        return bestParentIndex;
    }

// Rewire the tree to optimize paths (RRT* specific) - non-recursive version
void rewireTree(std::vector<Node>& nodes, int newNodeIdx, 
    const std::vector<int>& nearIndices,
    const std::vector<std::vector<double>>& obstacles) {
const Node& newNode = nodes[newNodeIdx];

// First pass: directly rewire neighbors of the new node
for (int nearIdx : nearIndices) {

    // Skip the parent of the new node
    if (nearIdx == nodes[newNodeIdx].parent) {
        continue;
    }

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
      // Check for potential cycle
      int tempParent = newNodeIdx;
      bool wouldCreateCycle = false;
      while (tempParent != -1) {
          if (tempParent == nearIdx) {
              wouldCreateCycle = true;
              break;
          }
          tempParent = nodes[tempParent].parent;
      }
      
      if (!wouldCreateCycle) {
          nodes[nearIdx].cost = costThroughNew;
          nodes[nearIdx].parent = newNodeIdx;
      }
  }
}
}

// Second pass: non-recursive descendant cost update
// Keep track of nodes that have been updated
std::vector<bool> updated(nodes.size(), false);
std::queue<int> nodesToUpdate;

// Add all near nodes that were rewired to the queue
for (int nearIdx : nearIndices) {
if (nearIdx != nodes[newNodeIdx].parent && 
  nodes[nearIdx].parent == newNodeIdx) {
  nodesToUpdate.push(nearIdx);
  updated[nearIdx] = true;
}
}

// Process queue until empty
while (!nodesToUpdate.empty()) {
int currentIdx = nodesToUpdate.front();
nodesToUpdate.pop();

// Find children of current node
for (size_t i = 0; i < nodes.size(); i++) {
  if (nodes[i].parent == currentIdx) {
      // Update child cost
      double newCost = nodes[currentIdx].cost + distance(nodes[currentIdx], nodes[i]);
      
      // Only update if cost improves
      if (newCost < nodes[i].cost) {
          // Check for potential cycle before updating
            int tempParent = currentIdx;
            bool wouldCreateCycle = false;
            while (tempParent != -1) {
                if (tempParent == i) {
                    wouldCreateCycle = true;
                    break;
                }
                tempParent = nodes[tempParent].parent;
            }
            
            if (!wouldCreateCycle) {
                nodes[i].cost = newCost;
                
                // Add to queue if not already updated
                if (!updated[i]) {
                    nodesToUpdate.push(i);
                    updated[i] = true;
                }
            }
      }
  }
}
}
}

// Main RRT* algorithm with option to stop at first solution
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
    bool enableVisualization,
    bool stopAtFirstSolution  // New parameter
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
        //std::cout << "Iteration " << i << "/" << maxIterations << std::endl;

        // Get current time for this iteration
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        double timeSeconds = elapsed.count();
        
        // Generate random node (with small probability, sample the goal)
        Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.05) ? 
                          goal : Node(xDist(gen), yDist(gen));
        
        // Find nearest node
        //std::cout << "Iteration " << i << ": About to call findNearest" << std::endl;
        int nearestIndex = findNearest(nodes, randomNode);
        //std::cout << "Iteration " << i << ": Finished findNearest" << std::endl;
        
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
                //std::cout << "Iteration " << i << ": About to call rewireTree" << std::endl;
                rewireTree(nodes, newNodeIndex, nearIndices, obstacles);
                //std::cout << "Iteration " << i << ": Finished rewireTree" << std::endl;

                // Check if we can reach the goal from this new node
                double distToGoal = distance(newNode, goal);
                if (distToGoal <= goalThreshold) {
                    //std::cout << "Iteration " << i << ": Found goal!" << std::endl;

                    // Check if path to goal is collision-free
                    if (isPathClear(newNode, goal, obstacles)) {
                        // Option 1: Continue searching for the optimal path until max iterations
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
                                int tempParent = newNodeIndex;
                                bool wouldCreateCycle = false;
                                while (tempParent != -1) {
                                    if (tempParent == goalNodeIndex) {
                                        wouldCreateCycle = true;
                                        break;
                                    }
                                    tempParent = nodes[tempParent].parent;
                                }
                                
                                if (!wouldCreateCycle) {
                                    // Replace existing goal node with better path
                                    nodes[goalNodeIndex] = goalNode;
                                } else {
                                    std::cout << "Avoided cycle when updating goal node" << std::endl;
                                }
                            }
                            
                            // Option 2: If we want to stop at first solution
                            if (stopAtFirstSolution) {
                                // Save the tree data if visualization is enabled
                                if (enableVisualization) {
                                    saveTreeToFile(nodes, treeFilename);
                                }
                                
                                std::cout << "Goal reached in " << i << " iterations. Stopping search." << std::endl;
                                
                                // Extract and return path
                                return extractPath(nodes, goalNodeIndex);
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

} // end of using namespace