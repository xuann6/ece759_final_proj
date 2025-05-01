#include "rrtBidirectionalWithObstacles.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace rrt_bidirectional_obstacles {

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

// Steer from nearest node towards random node with a maximum step size
Node steer(const Node& nearest, const Node& random, double stepSize) {
    double dist = distance(nearest, random);
    
    if (dist <= stepSize) {
        return Node(random.x, random.y);
    } else {
        double ratio = stepSize / dist;
        double newX = nearest.x + ratio * (random.x - nearest.x);
        double newY = nearest.y + ratio * (random.y - nearest.y);
        return Node(newX, newY);
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

// Try to connect the two trees
bool tryConnect(const std::vector<Node>& treeA, 
                std::vector<Node>& treeB, 
                int nearestIndexA, 
                Node& newNodeB, 
                double stepSize,
                const std::vector<Obstacle>& obstacles) {
    // Get the node from tree A that we're trying to connect to
    const Node& nodeA = treeA[nearestIndexA];
    
    // Check direct connection (if distance is small enough)
    if (distance(nodeA, newNodeB) <= stepSize) {
        // Check for collision
        if (!checkCollision(nodeA, newNodeB, obstacles)) {
            // Trees can be connected directly
            return true;
        }
    }
    
    // We can't connect directly, try stepping towards nodeA
    Node steppedNode = steer(newNodeB, nodeA, stepSize);
    
    // Check for collision
    if (checkCollision(newNodeB, steppedNode, obstacles)) {
        return false; // Can't step towards nodeA
    }
    
    // Update newNodeB to the stepped position
    newNodeB = steppedNode;
    return false; // Trees are not connected yet, but stepped towards connection
}

// Extract path from start to goal by traversing both trees
std::vector<Node> extractPath(const std::vector<Node>& startTree, 
                             const std::vector<Node>& goalTree,
                             int startConnectIndex,
                             int goalConnectIndex) {
    std::vector<Node> path;
    
    // Extract path from start tree (from start to connection point)
    std::vector<Node> startPath;
    int currentIndex = startConnectIndex;
    
    while (currentIndex != -1) {
        startPath.push_back(startTree[currentIndex]);
        currentIndex = startTree[currentIndex].parent;
    }
    
    // Reverse start path since we need it from start to connection point
    std::reverse(startPath.begin(), startPath.end());
    
    // Extract path from goal tree (from connection point to goal)
    std::vector<Node> goalPath;
    currentIndex = goalConnectIndex;
    
    while (currentIndex != -1) {
        goalPath.push_back(goalTree[currentIndex]);
        currentIndex = goalTree[currentIndex].parent;
    }
    
    // Combine the paths
    path = startPath;
    path.insert(path.end(), goalPath.begin(), goalPath.end());
    
    return path;
}

// Main Bidirectional RRT algorithm with obstacle avoidance
BiRRTResult buildBidirectionalRRTWithObstacles(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize,
    double connectThreshold,
    int maxIterations,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    const std::string& startTreeFilename,
    const std::string& goalTreeFilename,
    bool enableVisualization
) {
    BiRRTResult result;
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> xDist(xMin, xMax);
    std::uniform_real_distribution<> yDist(yMin, yMax);
    
    // Check if start or goal is inside an obstacle
    for (const auto& obstacle : obstacles) {
        if (obstacle.contains(start.x, start.y)) {
            std::cout << "Start position is inside an obstacle!" << std::endl;
            return result;
        }
        if (obstacle.contains(goal.x, goal.y)) {
            std::cout << "Goal position is inside an obstacle!" << std::endl;
            return result;
        }
    }
    
    // Initialize trees
    std::vector<Node> startTree;
    std::vector<Node> goalTree;
    
    startTree.push_back(start); // Start tree with start node
    goalTree.push_back(goal);   // Goal tree with goal node
    
    // Connection points between the trees
    int startConnectIndex = -1;
    int goalConnectIndex = -1;
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Alternate growing trees (grow start tree on even iterations, goal tree on odd)
        bool growingStartTree = (i % 2 == 0);
        std::vector<Node>& activeTree = growingStartTree ? startTree : goalTree;
        std::vector<Node>& targetTree = growingStartTree ? goalTree : startTree;
        
        // Generate random node
        Node randomNode(xDist(gen), yDist(gen));
        
        // Skip if random node is inside an obstacle
        bool insideObstacle = false;
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(randomNode.x, randomNode.y)) {
                insideObstacle = true;
                break;
            }
        }
        if (insideObstacle) continue;
        
        // Find nearest node in the active tree
        int nearestActiveIndex = findNearest(activeTree, randomNode);
        if (nearestActiveIndex < 0) continue; // Skip if tree is empty
        
        // Steer towards the random node
        Node newNode = steer(activeTree[nearestActiveIndex], randomNode, stepSize);
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
        if (checkCollision(activeTree[nearestActiveIndex], newNode, obstacles)) {
            continue; // Skip if there's a collision
        }
        
        // Add the node to the active tree
        newNode.parent = nearestActiveIndex;
        activeTree.push_back(newNode);
        int newNodeIndex = activeTree.size() - 1;
        
        // Find the nearest node in the target tree
        int nearestTargetIndex = findNearest(targetTree, newNode);
        if (nearestTargetIndex < 0) continue; // Skip if target tree is empty
        
        // Check if we can connect the trees
        if (distance(newNode, targetTree[nearestTargetIndex]) <= connectThreshold) {
            // Check for collision between connection points
            if (!checkCollision(newNode, targetTree[nearestTargetIndex], obstacles)) {
                // Trees are connected!
                if (growingStartTree) {
                    startConnectIndex = newNodeIndex;
                    goalConnectIndex = nearestTargetIndex;
                } else {
                    startConnectIndex = nearestTargetIndex;
                    goalConnectIndex = newNodeIndex;
                }
                
                // Set iteration count in result
                result.iterations = i + 1;
                break;
            }
        }
        
        // Try to step towards the nearest node in the target tree
        Node connectNode = targetTree[nearestTargetIndex];
        bool connected = tryConnect(activeTree, targetTree, newNodeIndex, connectNode, stepSize, obstacles);
        
        if (connected) {
            // We successfully connected the trees
            if (growingStartTree) {
                startConnectIndex = newNodeIndex;
                goalConnectIndex = nearestTargetIndex;
            } else {
                startConnectIndex = nearestTargetIndex;
                goalConnectIndex = newNodeIndex;
            }
            
            // Set iteration count in result
            result.iterations = i + 1;
            break;
        }
    }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.executionTime = elapsed.count();
    
    // Set tree sizes in result
    result.startTreeSize = startTree.size();
    result.goalTreeSize = goalTree.size();
    
    // Save tree data for visualization
    if (enableVisualization) {
        // Save start tree
        std::ofstream startFile(startTreeFilename);
        if (startFile.is_open()) {
            startFile << "node_id,x,y,parent_id,time" << std::endl;
            for (size_t j = 0; j < startTree.size(); j++) {
                startFile << j << "," << startTree[j].x << "," << startTree[j].y << ","
                         << startTree[j].parent << "," << startTree[j].time << std::endl;
            }
            startFile.close();
            std::cout << "Start tree data saved to " << startTreeFilename << std::endl;
        }
        
        // Save goal tree
        std::ofstream goalFile(goalTreeFilename);
        if (goalFile.is_open()) {
            goalFile << "node_id,x,y,parent_id,time" << std::endl;
            for (size_t j = 0; j < goalTree.size(); j++) {
                goalFile << j << "," << goalTree[j].x << "," << goalTree[j].y << ","
                       << goalTree[j].parent << "," << goalTree[j].time << std::endl;
            }
            goalFile.close();
            std::cout << "Goal tree data saved to " << goalTreeFilename << std::endl;
        }
    }
    
    if (startConnectIndex != -1 && goalConnectIndex != -1) {
        // Trees were connected, extract the path
        result.path = extractPath(startTree, goalTree, startConnectIndex, goalConnectIndex);
        std::cout << "Path found with " << result.path.size() << " nodes using bidirectional RRT." << std::endl;
        std::cout << "Start tree: " << startTree.size() << " nodes, Goal tree: " << goalTree.size() << " nodes" << std::endl;
    } else {
        std::cout << "Maximum number of iterations reached without finding a path with bidirectional RRT." << std::endl;
    }
    
    return result;
}

} // namespace rrt_bidirectional_obstacles 