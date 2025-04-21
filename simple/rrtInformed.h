#ifndef RRT_INFORMED_H
#define RRT_INFORMED_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include "rrt.h"
#include "rrtStar.h"

namespace rrt_informed {

    // Structure to represent an ellipsoid sampling domain
    struct EllipsoidSamplingDomain {
        Node center;      // Center of the ellipsoid
        double transverse_axis; // Length of the major axis (cbest)
        double conjugate_axis;  // Length of the minor axis (sqrt(cbest^2 - cmin^2))
        double cmin;      // Minimum possible cost (Euclidean distance between start and goal)
        
        // Constructor - Initialize all values in the initializer list
        EllipsoidSamplingDomain(const Node& start, const Node& goal, double current_best_cost) 
            : center(Node((start.x + goal.x) / 2.0, (start.y + goal.y) / 2.0)),
              cmin(distance(start, goal)),
              transverse_axis(current_best_cost),
              conjugate_axis(0.0)  // Initialize to zero, then update in body
        {
            // Calculate conjugate axis length with proper error handling
            if (current_best_cost > cmin) {
                conjugate_axis = std::sqrt(current_best_cost * current_best_cost - cmin * cmin);
            }
        }
        
        // Check if the domain is valid (if not, we'll use the entire state space)
        bool isValid() const {
            return (transverse_axis > cmin) && (cmin > 0) && (conjugate_axis > 0);
        }
    };

    // Generate a random sample in the unit ball
    Node sampleUnitBall();
    
    // Transform a point from the unit ball to the ellipsoid
    Node transformToEllipsoid(const Node& ball_sample, const EllipsoidSamplingDomain& domain, 
                              const Node& start, const Node& goal);
    
    // Sample a state from the ellipsoidal domain
    Node sampleInformedSubset(const Node& start, const Node& goal, double cbest,
                              double xMin, double xMax, double yMin, double yMax);
    
    // Calculate the rotation matrix from the ellipsoid frame to the world frame
    std::vector<std::vector<double>> rotationToWorldFrame(const Node& start, const Node& goal);
    
    // Main Informed RRT* algorithm
    std::vector<Node> buildInformedRRTStar(
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
        const std::string& treeFilename = "rrt_informed_tree.csv",
        bool enableVisualization = true
    );

} // namespace rrt_informed

#endif // RRT_INFORMED_H