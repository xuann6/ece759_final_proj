#ifndef RRT_OBSTACLES_H
#define RRT_OBSTACLES_H

#include <iostream>
#include <vector>
#include <string>

// Structure to represent a node in the RRT
struct Node {
    double x, y;  // 2D position
    int parent;   // Index of parent node in the nodes vector
    double time;  // Time when node was added (for visualization)

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

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b);

// Find nearest node in the tree to the given point
int findNearest(const std::vector<Node>& nodes, const Node& point);

// Steer from nearest node towards random node with a maximum step size
Node steer(const Node& nearest, const Node& random, double stepSize);

// Check if a line segment between two nodes collides with any obstacle
bool checkCollision(const Node& a, const Node& b, const std::vector<Obstacle>& obstacles);

// Extract path from start to goal by traversing the tree backwards
std::vector<Node> extractPath(const std::vector<Node>& nodes, int goalIndex);

// Save the tree data to a file for visualization
void saveTreeToFile(const std::vector<Node>& nodes, const std::string& filename);

// Generate random obstacles in the environment
std::vector<Obstacle> generateObstacles(double worldWidth, double worldHeight);

// Main RRT algorithm with obstacle avoidance
std::vector<Node> buildRRTWithObstacles(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize = 0.1,
    double goalThreshold = 0.1,
    int maxIterations = 1000,
    double xMin = 0.0,
    double xMax = 1.0,
    double yMin = 0.0,
    double yMax = 1.0,
    const std::string& treeFilename = "rrt_obstacles_tree.csv",
    bool enableVisualization = true
);

#endif // RRT_OBSTACLES_H 