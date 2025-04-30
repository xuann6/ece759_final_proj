#ifndef RRT_STAR_OMP_H
#define RRT_STAR_OMP_H

#include <iostream>
#include <vector>
#include <string>
#include "../simple/rrt.h"
#include "rrtOmp.h"

namespace rrt_star_omp {

    // Find nodes within a certain radius in parallel
    std::vector<int> findNearNodesParallel(const std::vector<Node>& nodes, const Node& newNode, double radius);

    // Choose best parent for a new node (RRT* specific) in parallel
    int chooseBestParentParallel(const std::vector<Node>& nodes, const Node& newNode, 
                               const std::vector<int>& nearIndices, 
                               const std::vector<std::vector<double>>& obstacles);

    // Rewire the tree to optimize paths (RRT* specific) with parallel execution
    void rewireTreeParallel(std::vector<Node>& nodes, int newNodeIdx, 
                         const std::vector<int>& nearIndices,
                         const std::vector<std::vector<double>>& obstacles);
    
    // Main Parallel RRT* algorithm
    std::vector<Node> buildRRTStarOmp(
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
        const std::string& treeFilename = "rrt_star_omp_tree.csv",
        bool enableVisualization = true,
        int numThreads = 4
    );

} // namespace rrt_star_omp

#endif // RRT_STAR_OMP_H