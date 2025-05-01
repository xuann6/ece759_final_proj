#include "rrtStarWithObstacles.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace rrt_star_obstacles {

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Find nearest node in the tree to the given point
int findNearest(const std::vector<Node>& nodes, const Node& point) {
    if (nodes.empty()) return -1;
    
    int nearest = 0;
    double minDist = distance(nodes[0], point);
    
    for (size_t i = 1; i < nodes.size(); i++) {
        double dist = distance(nodes[i], point);
        if (dist < minDist) {
            minDist = dist;
            nearest = static_cast<int>(i);
        }
    }
    
    return nearest;
}

// Find nodes within a certain radius of a point
std::vector<int> findNodesInRadius(const std::vector<Node>& nodes, const Node& point, double radius) {
    std::vector<int> neighbors;
    
    for (size_t i = 0; i < nodes.size(); i++) {
        if (distance(nodes[i], point) <= radius) {
            neighbors.push_back(static_cast<int>(i));
        }
    }
    
    return neighbors;
}

// Steer from nearest node towards random node with a maximum step size
Node steer(const Node& nearest, const Node& random, double stepSize) {
    double dist = distance(nearest, random);
    
    if (dist <= stepSize) {
        return Node(random.x, random.y, -1, 0.0);
    } else {
        double ratio = stepSize / dist;
        double newX = nearest.x + ratio * (random.x - nearest.x);
        double newY = nearest.y + ratio * (random.y - nearest.y);
        return Node(newX, newY, -1, 0.0);
    }
}

// Check if a line segment between two nodes collides with any obstacle
bool checkCollision(const Node& a, const Node& b, const std::vector<Obstacle>& obstacles) {
    // Number of point samples along the line segment for collision checking
    const int numSamples = 10;
    
    for (int i = 0; i <= numSamples; i++) {
        // Interpolate between nodes a and b
        double t = static_cast<double>(i) / numSamples;
        double x = a.x + t * (b.x - a.x);
        double y = a.y + t * (b.y - a.y);
        
        // Check if point is inside any obstacle
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(x, y)) {
                return true; // Collision detected
            }
        }
    }
    
    return false; // No collision
}

// Calculate the cost of a new node
double calculateCost(const std::vector<Node>& nodes, int parentIndex, const Node& newNode) {
    return nodes[parentIndex].cost + distance(nodes[parentIndex], newNode);
}

// Update the parent of a node if a better path is found
bool updateParent(std::vector<Node>& nodes, int nodeIndex, const std::vector<int>& neighbors, const std::vector<Obstacle>& obstacles) {
    double currentCost = nodes[nodeIndex].cost;
    int bestParent = nodes[nodeIndex].parent;
    double bestCost = currentCost;
    
    for (int neighbor : neighbors) {
        if (neighbor != nodeIndex && neighbor != nodes[nodeIndex].parent) {
            double potentialCost = nodes[neighbor].cost + distance(nodes[neighbor], nodes[nodeIndex]);
            
            if (potentialCost < bestCost && !checkCollision(nodes[neighbor], nodes[nodeIndex], obstacles)) {
                bestCost = potentialCost;
                bestParent = neighbor;
            }
        }
    }
    
    if (bestParent != nodes[nodeIndex].parent) {
        nodes[nodeIndex].parent = bestParent;
        nodes[nodeIndex].cost = bestCost;
        return true;
    }
    
    return false;
}

// Rewire the tree to maintain optimality
void rewireTree(std::vector<Node>& nodes, int newNodeIndex, const std::vector<int>& neighbors, const std::vector<Obstacle>& obstacles) {
    for (int neighbor : neighbors) {
        if (neighbor != newNodeIndex && neighbor != nodes[newNodeIndex].parent) {
            double potentialCost = nodes[newNodeIndex].cost + distance(nodes[newNodeIndex], nodes[neighbor]);
            
            if (potentialCost < nodes[neighbor].cost && !checkCollision(nodes[newNodeIndex], nodes[neighbor], obstacles)) {
                nodes[neighbor].parent = newNodeIndex;
                nodes[neighbor].cost = potentialCost;
                
                // Recursively update costs of children
                for (size_t i = 0; i < nodes.size(); i++) {
                    if (nodes[i].parent == neighbor) {
                        nodes[i].cost = nodes[neighbor].cost + distance(nodes[neighbor], nodes[i]);
                    }
                }
            }
        }
    }
}

// Extract path from start to goal by traversing the tree backwards
std::vector<Node> extractPath(const std::vector<Node>& nodes, int goalIndex) {
    std::vector<Node> path;
    int currentIndex = goalIndex;
    
    while (currentIndex != -1) {
        path.push_back(nodes[currentIndex]);
        currentIndex = nodes[currentIndex].parent;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

// Main RRT* algorithm with obstacle avoidance
std::vector<Node> buildRRTStarWithObstacles(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize,
    double goalThreshold,
    int maxIterations,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    double maxNeighborDist,
    const std::string& treeFilename,
    bool enableVisualization
) {
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> xDist(xMin, xMax);
    std::uniform_real_distribution<> yDist(yMin, yMax);
    
    // Check if start or goal is inside an obstacle
    for (const auto& obstacle : obstacles) {
        if (obstacle.contains(start.x, start.y)) {
            std::cout << "Start position is inside an obstacle!" << std::endl;
            return std::vector<Node>();
        }
        if (obstacle.contains(goal.x, goal.y)) {
            std::cout << "Goal position is inside an obstacle!" << std::endl;
            return std::vector<Node>();
        }
    }
    
    // Initialize tree with start node
    std::vector<Node> nodes;
    nodes.push_back(start); // Start node at time 0, with cost 0
    
    // Best solution so far
    int bestGoalIndex = -1;
    double bestGoalCost = std::numeric_limits<double>::max();
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Generate random node (with 5% probability, sample the goal)
        Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.05) ? 
                          goal : Node(xDist(gen), yDist(gen));
        
        // Skip if random node is inside an obstacle
        bool insideObstacle = false;
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(randomNode.x, randomNode.y)) {
                insideObstacle = true;
                break;
            }
        }
        if (insideObstacle) continue;
        
        // Find nearest node in the tree
        int nearestIndex = findNearest(nodes, randomNode);
        if (nearestIndex < 0) continue; // Skip if tree is empty
        
        // Steer towards the random node
        Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
        newNode.time = static_cast<double>(i) / maxIterations; // Normalized time
        
        // Skip if new node is inside an obstacle
        insideObstacle = false;
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(newNode.x, newNode.y)) {
                insideObstacle = true;
                break;
            }
        }
        if (insideObstacle) continue;
        
        // Check for collision between nearest node and new node
        if (checkCollision(nodes[nearestIndex], newNode, obstacles)) {
            continue; // Skip if there's a collision
        }
        
        // Find nodes in neighborhood
        std::vector<int> neighbors = findNodesInRadius(nodes, newNode, maxNeighborDist);
        
        // Find the best parent for the new node
        int bestParent = nearestIndex;
        double bestCost = nodes[nearestIndex].cost + distance(nodes[nearestIndex], newNode);
        
        for (int neighbor : neighbors) {
            double potentialCost = nodes[neighbor].cost + distance(nodes[neighbor], newNode);
            
            if (potentialCost < bestCost && !checkCollision(nodes[neighbor], newNode, obstacles)) {
                bestCost = potentialCost;
                bestParent = neighbor;
            }
        }
        
        // Add the node to the tree with best parent
        newNode.parent = bestParent;
        newNode.cost = bestCost;
        nodes.push_back(newNode);
        int newNodeIndex = nodes.size() - 1;
        
        // Rewire the tree
        rewireTree(nodes, newNodeIndex, neighbors, obstacles);
        
        // Check if we reached the goal
        if (distance(newNode, goal) <= goalThreshold) {
            double goalCost = newNode.cost + distance(newNode, goal);
            
            if (bestGoalIndex == -1 || goalCost < bestGoalCost) {
                // This is a better path to goal
                bestGoalCost = goalCost;
                
                // Add the goal node to the tree
                Node finalNode = goal;
                finalNode.parent = newNodeIndex;
                finalNode.cost = goalCost;
                finalNode.time = 1.0; // Goal reached at final time
                
                // Replace any existing goal node or add a new one
                if (bestGoalIndex == -1) {
                    nodes.push_back(finalNode);
                    bestGoalIndex = nodes.size() - 1;
                } else {
                    nodes[bestGoalIndex] = finalNode;
                }
                
                // std::cout << "Found a path with cost: " << goalCost << std::endl;
            }
        }
    }
    
    // Save the tree data to file for visualization
    if (enableVisualization) {
        std::ofstream file(treeFilename);
        if (file.is_open()) {
            file << "node_id,x,y,parent_id,cost,time" << std::endl;
            for (size_t j = 0; j < nodes.size(); j++) {
                file << j << ","
                     << nodes[j].x << ","
                     << nodes[j].y << ","
                     << nodes[j].parent << ","
                     << nodes[j].cost << ","
                     << nodes[j].time << std::endl;
            }
            file.close();
            std::cout << "Tree data saved to " << treeFilename << std::endl;
        }
    }
    
    if (bestGoalIndex != -1) {
        return extractPath(nodes, bestGoalIndex);
    } else {
        std::cout << "Maximum number of iterations reached without finding a path." << std::endl;
        return std::vector<Node>();
    }
}

} // namespace rrt_star_obstacles 