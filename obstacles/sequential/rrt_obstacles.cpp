#include "rrt_obstacles.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Find nearest node in the tree to the given point
int findNearest(const std::vector<Node>& nodes, const Node& point) {
    int nearest = 0;
    double minDist = distance(nodes[0], point);
    
    for (int i = 1; i < nodes.size(); i++) {
        double dist = distance(nodes[i], point);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }
    
    return nearest;
}

// Steer from nearest node towards random node with a maximum step size
Node steer(const Node& nearest, const Node& random, double stepSize) {
    double dist = distance(nearest, random);
    
    if (dist <= stepSize) {
        return random;
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

// Save the tree data to a file for visualization
void saveTreeToFile(const std::vector<Node>& nodes, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "node_id,x,y,parent_id,time" << std::endl;
    
    // Write node data
    for (int i = 0; i < nodes.size(); i++) {
        file << i << ","
             << nodes[i].x << ","
             << nodes[i].y << ","
             << nodes[i].parent << ","
             << nodes[i].time << std::endl;
    }
    
    file.close();
    std::cout << "Tree data saved to " << filename << std::endl;
}

// Generate random obstacles in the environment
std::vector<Obstacle> generateObstacles(double worldWidth, double worldHeight) {
    std::vector<Obstacle> obstacles;
    
    // Create two rectangular obstacles with fixed width (1/10th of world width)
    // and fixed height (60% of world height)
    double obstacleWidth = worldWidth / 10.0;
    
    // First obstacle at 1/3 of the world width
    double x1 = worldWidth / 3.0 - obstacleWidth / 2.0;
    double height1 = 0.6 * worldHeight;
    obstacles.push_back(Obstacle(x1, 0, obstacleWidth, height1));
    
    // Second obstacle at 2/3 of the world width
    double x2 = 2.0 * worldWidth / 3.0 - obstacleWidth / 2.0;
    double height2 = 0.6 * worldHeight;
    obstacles.push_back(Obstacle(x2, worldHeight - height2, obstacleWidth, height2));
    
    return obstacles;
}

// Save obstacles to a file for visualization
void saveObstaclesToFile(const std::vector<Obstacle>& obstacles, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "x,y,width,height" << std::endl;
    
    // Write obstacle data
    for (const auto& obstacle : obstacles) {
        file << obstacle.x << ","
             << obstacle.y << ","
             << obstacle.width << ","
             << obstacle.height << std::endl;
    }
    
    file.close();
    std::cout << "Obstacle data saved to " << filename << std::endl;
}

// Main RRT algorithm with obstacle avoidance
std::vector<Node> buildRRTWithObstacles(
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
    const std::string& treeFilename,
    bool enableVisualization
) {
    // Start timing for all runs
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
    
    // Save obstacles to file if visualization is enabled
    if (enableVisualization) {
        saveObstaclesToFile(obstacles, "rrt_obstacles.csv");
    }
    
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
    nodes.push_back(Node(start.x, start.y, -1, 0.0)); // Start node at time 0
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Get current time for this iteration
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        double timeSeconds = elapsed.count();
        
        // Generate random node (with small probability, sample the goal)
        Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.1) ? 
                          goal : Node(xDist(gen), yDist(gen));
        
        // Verify that the random node is not inside an obstacle
        bool insideObstacle = false;
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(randomNode.x, randomNode.y)) {
                insideObstacle = true;
                break;
            }
        }
        
        // Skip this iteration if the random node is inside an obstacle
        if (insideObstacle) {
            continue;
        }
        
        // Find nearest node
        int nearestIndex = findNearest(nodes, randomNode);
        
        // Create new node by steering
        Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
        
        // Check if path to new node collides with any obstacle
        if (checkCollision(nodes[nearestIndex], newNode, obstacles)) {
            continue; // Skip this node if there's a collision
        }
        
        // Set parent and time
        newNode.parent = nearestIndex;
        newNode.time = timeSeconds;
        
        // Add new node to tree
        nodes.push_back(newNode);
        int newNodeIndex = nodes.size() - 1;
        
        // Check if goal reached
        if (distance(newNode, goal) <= goalThreshold) {
            // Check if path to goal is collision-free
            if (!checkCollision(newNode, goal, obstacles)) {
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
    }
    
    // If goal not reached, save tree anyway (if visualization is enabled)
    if (enableVisualization) {
        saveTreeToFile(nodes, treeFilename);
    }
    
    // If goal not reached, return empty path
    std::cout << "Goal not reached within max iterations." << std::endl;
    return std::vector<Node>();
} 