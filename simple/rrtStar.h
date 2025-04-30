#ifndef RRT_STAR_H
#define RRT_STAR_H

#include <iostream>
#include <vector>
#include <string>
#include "rrt.h"
namespace rrt_star {
 
    // Already defined in rrt.h, don't have to define here..
    // Structure to represent a node in the RRT*
    // struct Node {
    //    double x, y;   // 2D position
    //     int parent;    // Index of parent node in the nodes vector
    //     double time;   // Time when node was added (for visualization)
    //     double cost;   // Cost from start to this node (RRT* specific)
        
    //     Node(double x, double y, int parent = -1, double time = 0.0, double cost = 0.0)
    //         : x(x), y(y), parent(parent), time(time), cost(cost) {}
    //};

    // Calculate Euclidean distance between two nodes
    // double distance(const Node& a, const Node& b);

    // Find nearest node in the tree to the given point
    // int findNearest(const std::vector<Node>& nodes, const Node& point);

    // Steer from nearest node towards random node with a maximum step size
    // Node steer(const Node& nearest, const Node& random, double stepSize);


    // Find nodes within a certain radius
    std::vector<int> findNearNodes(const std::vector<Node>& nodes, const Node& newNode, double radius);

    // Extract path from start to goal by traversing the tree backwards
    // std::vector<Node> extractPath(const std::vector<Node>& nodes, int goalIndex);

    // Save the tree data to a file for visualization
    // void saveTreeToFile(const std::vector<Node>& nodes, const std::string& filename);

    // The following 3 functions are specific for RRT Star
    // Choose best parent for a new node (RRT* specific)
    int chooseBestParent(const std::vector<Node>& nodes, const Node& newNode, const std::vector<int>& nearIndices, const std::vector<std::vector<double>>& obstacles);

    // Rewire the tree to optimize paths (RRT* specific)
    void rewireTree(std::vector<Node>& nodes, int newNodeIdx, const std::vector<int>& nearIndices, const std::vector<std::vector<double>>& obstacles);

    // Check if path between two nodes is collision-free
    bool isPathClear(const Node& from, const Node& to, const std::vector<std::vector<double>>& obstacles);

    // Main RRT* algorithm
    std::vector<Node> buildRRTStar(
        const Node& start,
        const Node& goal,
        const std::vector<std::vector<double>>& obstacles,
        double stepSize = 0.1,
        double goalThreshold = 0.1,
        int maxIterations = 1000,
        double rewireRadius = 0.5,
        double xMin = 0.0,
        double xMax = 1.0,
        double yMin = 0.0,
        double yMax = 1.0,
        const std::string& treeFilename = "rrt_star_tree.csv",
        bool enableVisualization = true,
        bool stopAtFirstSolution = false
    );
}
#endif // RRT_STAR_H