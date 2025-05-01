#include "rrtOmpWithObstacles.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

namespace rrt_omp_obstacles {

// Calculate Euclidean distance between two nodes
double distance(const Node& a, const Node& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Find nearest node in the tree to the given point (parallel version)
int findNearestParallel(const std::vector<Node>& nodes, const Node& point) {
    int nearest = 0;
    double minDist = std::numeric_limits<double>::max();
    
    if (!nodes.empty()) {
        minDist = distance(nodes[0], point);
    } else {
        return -1; // Empty tree
    }
    
    #pragma omp parallel
    {
        int local_nearest = 0;
        double local_minDist = minDist;
        
        #pragma omp for nowait
        for (size_t i = 0; i < nodes.size(); i++) {
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
        return Node(random.x, random.y);
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
                break;
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

// Generate random obstacles in the environment
std::vector<Obstacle> generateObstacles(double worldWidth, double worldHeight) {
    std::vector<Obstacle> obstacles;
    
    // Create two rectangular obstacles with fixed width and height
    double obstacleWidth = worldWidth / 10.0;
    
    // First obstacle at 1/3 of the world width with fixed height (60% of world height)
    double x1 = worldWidth / 3.0 - obstacleWidth / 2.0;
    double height1 = 0.6 * worldHeight;
    obstacles.push_back(Obstacle(x1, 0, obstacleWidth, height1));
    
    // Second obstacle at 2/3 of the world width with fixed height (60% of world height)
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
        file << obstacle.x << "," << obstacle.y << ","
             << obstacle.width << "," << obstacle.height << std::endl;
    }
    
    file.close();
    std::cout << "Obstacle data saved to " << filename << std::endl;
}

// Main RRT algorithm with obstacle avoidance (OpenMP version)
std::vector<Node> buildRRTOmpWithObstacles(
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
    
    // Save obstacles to file if visualization is enabled
    if (enableVisualization) {
        saveObstaclesToFile(obstacles, "rrt_omp_obstacles.csv");
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
    nodes.push_back(start); // Start node at time 0
    
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
        int nearestIndex = findNearestParallel(nodes, randomNode);
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
        if (checkCollisionParallel(nodes[nearestIndex], newNode, obstacles, numThreads)) {
            continue; // Skip if there's a collision
        }
        
        // Add the node to the tree
        newNode.parent = nearestIndex;
        nodes.push_back(newNode);
        
        // Check if we reached the goal
        if (distance(newNode, goal) <= goalThreshold) {
            // Add the goal node to the tree
            Node finalNode = goal;
            finalNode.parent = nodes.size() - 1;
            finalNode.time = 1.0; // Goal reached at final time
            nodes.push_back(finalNode);
            
            if (enableVisualization) {
                // Save the tree data to file for visualization
                std::ofstream file(treeFilename);
                if (file.is_open()) {
                    file << "node_id,x,y,parent_id,time" << std::endl;
                    for (size_t j = 0; j < nodes.size(); j++) {
                        file << j << "," << nodes[j].x << "," << nodes[j].y << ","
                             << nodes[j].parent << "," << nodes[j].time << std::endl;
                    }
                    file.close();
                    std::cout << "Tree data saved to " << treeFilename << std::endl;
                }
            }
            
            return extractPath(nodes, nodes.size() - 1);
        }
    }
    
    std::cout << "Maximum number of iterations reached without finding a path." << std::endl;
    
    if (enableVisualization) {
        // Save the tree data to file for visualization even if no path was found
        std::ofstream file(treeFilename);
        if (file.is_open()) {
            file << "node_id,x,y,parent_id,time" << std::endl;
            for (size_t j = 0; j < nodes.size(); j++) {
                file << j << "," << nodes[j].x << "," << nodes[j].y << ","
                     << nodes[j].parent << "," << nodes[j].time << std::endl;
            }
            file.close();
            std::cout << "Tree data saved to " << treeFilename << std::endl;
        }
    }
    
    return std::vector<Node>();
}

} // namespace rrt_omp_obstacles 