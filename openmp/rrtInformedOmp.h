#ifndef RRT_INFORMED_OMP_H
#define RRT_INFORMED_OMP_H

#include <iostream>
#include <vector>
#include <string>
#include "../simple/rrt.h"
#include "../simple/rrtInformed.h"
#include "rrtOmp.h"

namespace rrt_informed_omp {

    // Structure to represent an ellipsoidal sampling domain (same as in rrtInformed.h)
    using EllipsoidSamplingDomain = rrt_informed::EllipsoidSamplingDomain;

    // Generate a random sample in the unit ball in parallel
    Node sampleUnitBallParallel();
    
    // Calculate the rotation matrix from the ellipsoid frame to the world frame - parallel version
    std::vector<std::vector<double>> rotationToWorldFrameParallel(const Node& start, const Node& goal);
    
    // Sample a state from the ellipsoidal domain in parallel
    Node sampleInformedSubsetParallel(const Node& start, const Node& goal, double cbest,
                                     double xMin, double xMax, double yMin, double yMax);
    
    // Main Informed RRT* algorithm with OpenMP parallelization
    std::vector<Node> buildInformedRRTStarOmp(
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
        const std::string& treeFilename = "rrt_informed_omp_tree.csv",
        bool enableVisualization = true,
        int numThreads = 4,
        bool stopAtFirstSolution = false
    );

} // namespace rrt_informed_omp

#endif // RRT_INFORMED_OMP_H