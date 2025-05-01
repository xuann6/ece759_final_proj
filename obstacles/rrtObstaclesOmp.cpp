#include "rrtObstaclesOmp.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

namespace rrt_obstacles_omp {

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Find nearest node in the tree to the given point (parallel version)
int findNearestParallel(const std::vector<Node>& nodes, const Node& point) {
    int nearest = 0;
    double minDist = distance(nodes[0], point);
    
    #pragma omp parallel
    {
        int local_nearest = 0;
        double local_minDist = minDist;
        
        #pragma omp for nowait
        for (size_t i = 1; i < nodes.size(); i++) {
            double dist = distance(nodes[i], point);
            if (dist < local_minDist) {
                local_minDist = dist;
                local_nearest = static_cast<int>(i);
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

// Check if a line segment between two nodes collides with any obstacle (parallel version)
bool checkCollisionParallel(const Node& a, const Node& b, const std::vector<Obstacle>& obstacles, int numThreads) {
    // Number of point samples along the line segment for collision checking
    const int numSamples = 10;
    bool collision = false;
    
    // Set the number of OpenMP threads
    omp_set_num_threads(numThreads);
    
    #pragma omp parallel for shared(collision)
    for (int i = 0; i <= numSamples; i++) {
        // Skip if collision already detected
        if (collision) continue;
        
        // Interpolate between nodes a and b
        double t = static_cast<double>(i) / numSamples;
        double x = a.x + t * (b.x - a.x);
        double y = a.y + t * (b.y - a.y);
        
        // Check if point is inside any obstacle
        for (const auto& obstacle : obstacles) {
            if (obstacle.contains(x, y)) {
                #pragma omp atomic write
                collision = true;
            }
        }
    }
    
    return collision;
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
    for (size_t i = 0; i < nodes.size(); i++) {
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
    
    // Create two rectangular obstacles with fixed width and height
    double obstacleWidth = worldWidth / 10.0;
    
    // First obstacle at 1/3 of the world width with fixed height (40% of world height)
    double x1 = worldWidth / 3.0 - obstacleWidth / 2.0;
    double height1 = 0.6 * worldHeight;
    obstacles.push_back(Obstacle(x1, 0, obstacleWidth, height1));
    
    // Second obstacle at 2/3 of the world width with fixed height (40% of world height)
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

// Main RRT algorithm with obstacle avoidance (OpenMP version)
std::vector<Node> buildRRTWithObstaclesOmp(
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
    bool enableVisualization,
    int numThreads
) {
    // Set the number of OpenMP threads
    omp_set_num_threads(numThreads);
    
    // Start timing for all runs
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
    
    // Save obstacles to file if visualization is enabled
    if (enableVisualization) {
        saveObstaclesToFile(obstacles, "rrt_obstacles_omp.csv");
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
        
        #pragma omp parallel for shared(insideObstacle)
        for (size_t j = 0; j < obstacles.size(); j++) {
            if (insideObstacle) continue; // Skip if already inside an obstacle
            
            if (obstacles[j].contains(randomNode.x, randomNode.y)) {
                #pragma omp atomic write
                insideObstacle = true;
            }
        }
        
        if (insideObstacle) {
            continue; // Skip this random node if it's inside an obstacle
        }
        
        // Find nearest node in the tree
        int nearestIndex = findNearestParallel(nodes, randomNode);
        const Node& nearestNode = nodes[nearestIndex];
        
        // Steer towards random node with a maximum step size
        Node newNode = steer(nearestNode, randomNode, stepSize);
        newNode.parent = nearestIndex;
        newNode.time = timeSeconds;
        
        // Check if the path to the new node collides with any obstacle
        if (checkCollisionParallel(nearestNode, newNode, obstacles, numThreads)) {
            continue; // Skip this new node if path collides with an obstacle
        }
        
        // Add new node to the tree
        nodes.push_back(newNode);
        
        // Check if we reached the goal
        if (distance(newNode, goal) <= goalThreshold) {
            nodes.push_back(Node(goal.x, goal.y, nodes.size() - 1, timeSeconds));
            
            // Save the final tree for visualization if enabled
            if (enableVisualization) {
                saveTreeToFile(nodes, treeFilename);
            }
            
            // Extract the path from start to goal
            return extractPath(nodes, nodes.size() - 1);
        }
    }
    
    // Save the final tree for visualization if enabled
    if (enableVisualization) {
        saveTreeToFile(nodes, treeFilename);
    }
    
    // No path found within the maximum number of iterations
    return std::vector<Node>();
}

} // namespace rrt_obstacles_omp 