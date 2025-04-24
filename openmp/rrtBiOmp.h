#ifndef BIDIRECTIONAL_RRT_H
#define BIDIRECTIONAL_RRT_H

#include <iostream>
#include <vector>
#include <string>
#include "../simple/rrt.h"

namespace bidirectional_rrt_omp {

    // Find the closest pair of nodes between two trees
    std::pair<int, int> findClosestNodes(const std::vector<Node>& treeA, const std::vector<Node>& treeB);
    
    // Try to connect two trees at their closest nodes
    bool tryConnect(std::vector<Node>& treeA, std::vector<Node>& treeB, 
                    const std::vector<std::vector<double>>& obstacles,
                    double stepSize, double connectThreshold);
    
    // Extend a single tree towards a random point
    bool extendTree(std::vector<Node>& tree, const Node& randomNode, 
                    const std::vector<std::vector<double>>& obstacles,
                    double stepSize);
    
    // Check if goal has been reached (tree connection in bidirectional RRT)
    bool isGoalReached(const std::vector<Node>& startTree, const std::vector<Node>& goalTree, 
                       const std::vector<std::vector<double>>& obstacles,
                       double threshold);
    
    // Check if path between two nodes is collision-free
    bool isPathClear(const Node& from, const Node& to,
                     const std::vector<std::vector<double>>& obstacles);
    
    // Merge two trees to create a complete path
    std::vector<Node> mergeTrees(const std::vector<Node>& startTree, int startConnectIndex, 
                               const std::vector<Node>& goalTree, int goalConnectIndex);
    
    // Extract the final path from merged trees
    std::vector<Node> extractBidirectionalPath(const std::vector<Node>& mergedTree, int startIndex, int goalIndex);
    
    // Main Bidirectional RRT algorithm
    std::vector<Node> buildBidirectionalRRT(
        const Node& start,
        const Node& goal,
        const std::vector<std::vector<double>>& obstacles,
        double stepSize = 0.1,
        double connectThreshold = 0.1,
        int maxIterations = 1000,
        double xMin = 0.0,
        double xMax = 1.0,
        double yMin = 0.0,
        double yMax = 1.0,
        const std::string& treeFilename = "rrt_bidirectional_tree.csv",
        bool enableVisualization = true,
        int numThreads = 4
    );
    
    // Build partial path from both trees when connection fails
    std::vector<Node> buildPartialPath(
        const std::vector<Node>& startTree,
        const std::vector<Node>& goalTree,
        const std::vector<std::vector<double>>& obstacles
    );

    // Save both trees data to a single file for visualization
    void saveTreesToFile(
        const std::vector<Node>& startTree, 
        const std::vector<Node>& goalTree,
        const std::string& treeFilename
    );
}

#endif // BIDIRECTIONAL_RRT_H