#include "rrtInformedOmp.h"
#include "rrtStarOmp.h"
#include "../simple/rrtInformed.h"
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

namespace rrt_informed_omp {

    // Generate a random sample in the unit ball - parallel version
    // (This function is essentially the same as the non-parallel version since
    // it's a small operation that doesn't benefit much from parallelization)
    Node sampleUnitBallParallel() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        
        double x, y, mag_squared;
        do {
            x = dist(gen);
            y = dist(gen);
            mag_squared = x*x + y*y;
        } while (mag_squared > 1.0);
        
        return Node(x, y);
    }
    
    // Calculate the rotation matrix from the ellipsoid frame to the world frame - parallel version
    std::vector<std::vector<double>> rotationToWorldFrameParallel(const Node& start, const Node& goal) {
        // Get the direction of the transverse axis (from start to goal)
        double a1_x = goal.x - start.x;
        double a1_y = goal.y - start.y;
        double norm = std::sqrt(a1_x*a1_x + a1_y*a1_y);
        
        // Normalize
        a1_x /= norm;
        a1_y /= norm;
        
        // Get the orthogonal direction
        double a2_x = -a1_y;
        double a2_y = a1_x;
        
        // Create rotation matrix [a1 a2]
        std::vector<std::vector<double>> C(2, std::vector<double>(2));
        C[0][0] = a1_x; C[0][1] = a2_x;
        C[1][0] = a1_y; C[1][1] = a2_y;
        
        return C;
    }
    
    // Transform a point from the unit ball to the ellipsoid - parallel version
    Node transformToEllipsoidParallel(const Node& ball_sample, const EllipsoidSamplingDomain& domain,
                                    const Node& start, const Node& goal) {
        // Get the rotation matrix
        auto C = rotationToWorldFrameParallel(start, goal);
        
        // Diagonal scaling matrix (r1, r2)
        double r1 = domain.transverse_axis / 2.0;  // Transverse radius
        double r2 = domain.conjugate_axis / 2.0;   // Conjugate radius
        
        // Transform from ball to ellipsoid: x_ellipse = C * L * x_ball + center
        double x_trans = r1 * ball_sample.x;
        double y_trans = r2 * ball_sample.y;
        
        // Apply rotation
        double x_world = C[0][0] * x_trans + C[0][1] * y_trans + domain.center.x;
        double y_world = C[1][0] * x_trans + C[1][1] * y_trans + domain.center.y;
        
        return Node(x_world, y_world);
    }
    
    // Sample a state from the ellipsoidal domain or the entire state space - parallel version
    Node sampleInformedSubsetParallel(const Node& start, const Node& goal, double cbest,
                                   double xMin, double xMax, double yMin, double yMax) {
        // Create sampling domain
        EllipsoidSamplingDomain domain(start, goal, cbest);
        
        // If the domain is not valid or we haven't found a solution yet (cbest is infinite)
        if (!domain.isValid() || std::isinf(cbest)) {
            // Sample from the entire state space
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> x_dist(xMin, xMax);
            std::uniform_real_distribution<> y_dist(yMin, yMax);
            
            return Node(x_dist(gen), y_dist(gen));
        } else {
            // Sample from the ellipsoid
            Node ball_sample = sampleUnitBallParallel();
            Node ellipsoid_sample = transformToEllipsoidParallel(ball_sample, domain, start, goal);
            
            // Ensure the sample is within the state bounds
            double x = std::max(xMin, std::min(xMax, ellipsoid_sample.x));
            double y = std::max(yMin, std::min(yMax, ellipsoid_sample.y));
            
            return Node(x, y);
        }
    }

    // Main parallel Informed RRT* algorithm
    std::vector<Node> buildInformedRRTStarOmp(
        const Node& start,
        const Node& goal,
        const std::vector<std::vector<double>>& obstacles,
        double stepSize,
        double goalThreshold,
        int maxIterations,
        double rewireRadius,
        double xMin,
        double xMax,
        double yMin,
        double yMax,
        const std::string& treeFilename,
        bool enableVisualization,
        int numThreads,
        bool stopAtFirstSolution
    ) {
        // Set the number of OpenMP threads
        omp_set_num_threads(numThreads);
        
        // Start timing
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime = 
            std::chrono::high_resolution_clock::now();
        
        // Initialize tree with start node (cost = 0)
        std::vector<Node> nodes;
        nodes.push_back(Node(start.x, start.y, -1, 0.0, 0.0)); // Start node at time 0, cost 0
        
        // Random number generation setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xDist(xMin, xMax);
        std::uniform_real_distribution<> yDist(yMin, yMax);
        
        // Best solution found so far
        double bestCost = std::numeric_limits<double>::infinity();
        int goalNodeIndex = -1;
        
        // Main loop
        for (int i = 0; i < maxIterations; i++) {
            // Get current time for this iteration
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;
            double timeSeconds = elapsed.count();
            
            // Sample from the informed subset (ellipsoid) if we have a solution
            Node randomNode = sampleInformedSubsetParallel(start, goal, bestCost, xMin, xMax, yMin, yMax);
            
            // Find nearest node using parallel version
            int nearestIndex = rrt_omp::findNearestParallel(nodes, randomNode);
            
            // Create new node by steering
            Node newNode = steer(nodes[nearestIndex], randomNode, stepSize);
            newNode.time = timeSeconds;
            
            // Check if path to new node is collision-free
            if (rrt_star::isPathClear(nodes[nearestIndex], newNode, obstacles)) {
                // Find nodes within the rewiring radius (parallel)
                std::vector<int> nearIndices = rrt_star_omp::findNearNodesParallel(nodes, newNode, rewireRadius);
                
                // Choose best parent (parallel)
                int bestParentIndex = rrt_star_omp::chooseBestParentParallel(nodes, newNode, nearIndices, obstacles);
                
                if (bestParentIndex != -1) {
                    // Set parent and cost for the new node
                    newNode.parent = bestParentIndex;
                    newNode.cost = nodes[bestParentIndex].cost + distance(nodes[bestParentIndex], newNode);
                    
                    // Add new node to tree
                    nodes.push_back(newNode);
                    int newNodeIndex = nodes.size() - 1;
                    
                    // Rewire the tree (parallel)
                    rrt_star_omp::rewireTreeParallel(nodes, newNodeIndex, nearIndices, obstacles);
                    
                    // Check if we can reach the goal from this new node
                    double distToGoal = distance(newNode, goal);
                    if (distToGoal <= goalThreshold) {
                        // Check if path to goal is collision-free
                        if (rrt_star::isPathClear(newNode, goal, obstacles)) {
                            // Calculate total cost to goal
                            double totalCost = newNode.cost + distToGoal;
                            
                            // If this path is better than previous solutions
                            if (totalCost < bestCost) {
                                bestCost = totalCost;
                                
                                // Create goal node
                                Node goalNode = goal;
                                goalNode.parent = newNodeIndex;
                                goalNode.cost = totalCost;
                                
                                // Set time for goal node
                                auto goalTime = std::chrono::high_resolution_clock::now();
                                std::chrono::duration<double> goalElapsed = goalTime - startTime;
                                goalNode.time = goalElapsed.count();
                                
                                // Add or update goal node
                                if (goalNodeIndex == -1) {
                                    nodes.push_back(goalNode);
                                    goalNodeIndex = nodes.size() - 1;
                                    
                                    // Option to stop at first solution
                                    if (stopAtFirstSolution) {
                                        // Save the tree data if visualization is enabled
                                        if (enableVisualization) {
                                            saveTreeToFile(nodes, treeFilename);
                                        }
                                        
                                        std::cout << "Goal reached in " << i << " iterations. Stopping search." << std::endl;
                                        
                                        // Extract and return path
                                        return extractPath(nodes, nodes.size() - 1);
                                    }
                                } else {
                                    // Replace existing goal node with better path
                                    nodes[goalNodeIndex] = goalNode;
                                }
                                
                                // Log the improved solution
                                std::cout << "Improved solution found at iteration " << i 
                                          << " with cost: " << bestCost << std::endl;
                            }
                        }
                    }
                }
            }
            
            // Periodically save the tree for visualization
            if (enableVisualization && i % 100 == 0) {
                saveTreeToFile(nodes, treeFilename);
            }
        }
        
        // Save the final tree data if visualization is enabled
        if (enableVisualization) {
            saveTreeToFile(nodes, treeFilename);
        }
        
        // If goal was reached, extract and return the path
        if (goalNodeIndex != -1) {
            return extractPath(nodes, goalNodeIndex);
        } else {
            // If goal not reached, return empty path
            std::cout << "Goal not reached within max iterations." << std::endl;
            return std::vector<Node>();
        }
    }

} // end of namespace