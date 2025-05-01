#ifndef RRT_BIDIRECTIONAL_WITH_OBSTACLES_H
#define RRT_BIDIRECTIONAL_WITH_OBSTACLES_H

#include <iostream>
#include <vector>
#include <string>

namespace rrt_bidirectional_obstacles {

// Structure to represent a node in the RRT
struct Node {
    double x, y;       // 2D position
    int parent;        // Index of parent node in the nodes vector
    double time;       // Time when node was added (for visualization)

    Node(double x, double y, int parent = -1, double time = 0.0) 
        : x(x), y(y), parent(parent), time(time) {}
};

// Structure to represent a rectangular obstacle
struct Obstacle {
    double x, y;      // Bottom-left corner position
    double width, height;  // Dimensions
    
    Obstacle(double x, double y, double width, double height)
        : x(x), y(y), width(width), height(height) {}
        
    // Check if a point is inside the obstacle
    bool contains(double pointX, double pointY) const {
        return pointX >= x && pointX <= x + width &&
               pointY >= y && pointY <= y + height;
    }
};

// Structure to contain results of bidirectional search
struct BiRRTResult {
    std::vector<Node> path;   // Final path from start to goal
    int startTreeSize;        // Number of nodes in the start tree
    int goalTreeSize;         // Number of nodes in the goal tree
    int iterations;           // Number of iterations performed
    double executionTime;     // Total execution time
    
    BiRRTResult() : startTreeSize(0), goalTreeSize(0), iterations(0), executionTime(0.0) {}
};

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b);

// Find nearest node in the tree to the given point
int findNearest(const std::vector<Node>& nodes, const Node& point);

// Steer from nearest node towards random node with a maximum step size
Node steer(const Node& nearest, const Node& random, double stepSize);

// Check if a line segment between two nodes collides with any obstacle
bool checkCollision(const Node& a, const Node& b, const std::vector<Obstacle>& obstacles);

// Try to connect the two trees
bool tryConnect(const std::vector<Node>& treeA, 
                std::vector<Node>& treeB, 
                int nearestIndexA, 
                Node& newNodeB, 
                double stepSize,
                const std::vector<Obstacle>& obstacles);

// Extract path from start to goal by traversing both trees
std::vector<Node> extractPath(const std::vector<Node>& startTree, 
                             const std::vector<Node>& goalTree,
                             int startConnectIndex,
                             int goalConnectIndex);

// Main Bidirectional RRT algorithm with obstacle avoidance
BiRRTResult buildBidirectionalRRTWithObstacles(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize = 0.1,
    double connectThreshold = 0.1,
    int maxIterations = 1000,
    double xMin = 0.0,
    double xMax = 1.0,
    double yMin = 0.0,
    double yMax = 1.0,
    const std::string& startTreeFilename = "rrt_bidirectional_obstacles_start_tree.csv",
    const std::string& goalTreeFilename = "rrt_bidirectional_obstacles_goal_tree.csv",
    bool enableVisualization = true
);

} // namespace rrt_bidirectional_obstacles

#endif // RRT_BIDIRECTIONAL_WITH_OBSTACLES_H 