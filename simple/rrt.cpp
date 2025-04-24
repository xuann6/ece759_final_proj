#include "rrt.h"
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

// Extract path from start to goal by traversing the tree backwards
std::vector<Node> extractPath(const std::vector<Node>& nodes, int goalIndex) {
    //std::cout << "Entered extractPath with goalIndex: " << goalIndex << std::endl;
    std::vector<Node> path;
    int currentIndex = goalIndex;
    
    // Safety check to prevent infinite loops due to cycles
    std::vector<bool> visited(nodes.size(), false);
    
    while (currentIndex != -1) {
        // Check if we've already visited this node - signals a cycle
        if (visited[currentIndex]) {
            std::cout << "ERROR: Cycle detected in path at node " << currentIndex << std::endl;
            break;
        }
        
        visited[currentIndex] = true;
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

// Main RRT algorithm
std::vector<Node> buildRRT(
    const Node& start,
    const Node& goal,
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
    
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> xDist(xMin, xMax);
    std::uniform_real_distribution<> yDist(yMin, yMax);
    
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
        Node randomNode = (std::uniform_real_distribution<>(0, 1)(gen) < 0.05) ? 
                          goal : Node(xDist(gen), yDist(gen));
        
        // Find nearest node
        int nearestIndex = findNearest(nodes, randomNode);
        
        // Create new node by steering
        Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
        newNode.parent = nearestIndex;
        newNode.time = timeSeconds;
        
        // Add new node to tree
        nodes.push_back(newNode);
        int newNodeIndex = nodes.size() - 1;
        
        // Check if goal reached
        if (distance(newNode, goal) <= goalThreshold) {
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
    
    // If goal not reached, save tree anyway (if visualization is enabled)
    if (enableVisualization) {
        saveTreeToFile(nodes, treeFilename);
    }
    
    // If goal not reached, return empty path
    std::cout << "Goal not reached within max iterations." << std::endl;
    return std::vector<Node>();
}
