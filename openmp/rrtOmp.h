#ifndef RRT_OMP_H
#define RRT_OMP_H

#include <iostream>
#include <vector>
#include <string>
#include "../simple/rrt.h"

namespace rrt_omp {
    
    // Main RRT algorithm with OpenMP parallelization
    std::vector<Node> buildRRTOmp(
        const Node& start,
        const Node& goal,
        double stepSize = 0.1,
        double goalThreshold = 0.1,
        int maxIterations = 1000,
        double xMin = 0.0,
        double xMax = 1.0,
        double yMin = 0.0,
        double yMax = 1.0,
        const std::string& treeFilename = "rrt_omp_tree.csv",
        bool enableVisualization = true,
        int numThreads = 4
    );

    // Find nearest node in parallel
    int findNearestParallel(const std::vector<Node>& nodes, const Node& point);

} // namespace rrt_omp

#endif // RRT_OMP_H