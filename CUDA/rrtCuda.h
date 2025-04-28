// rrtCuda.h
#ifndef RRT_CUDA_H
#define RRT_CUDA_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "rrtCommon.h"  // Shared Node / Obstacle definitions

// Helper structure for CUDA device data
struct RRTCudaData {
    // Host vectors
    std::vector<Node> h_nodes;
    std::vector<Obstacle> h_obstacles;
    
    // Device arrays
    float* d_nodeX;       // Node x coordinates
    float* d_nodeY;       // Node y coordinates
    int* d_nodeParent;    // Node parent indices
    float* d_nodeTime;    // Node timestamps
    int h_nodeCount;      // Current number of nodes
    int d_nodeCapacity;   // Allocated capacity for nodes
    
    // Obstacle data on device
    float* d_obstacleX;
    float* d_obstacleY;
    float* d_obstacleWidth;
    float* d_obstacleHeight;
    int h_obstacleCount;
    
    // Random state for CUDA
    curandState* d_randStates;
    
    // Constructor
    RRTCudaData() : 
        d_nodeX(nullptr), d_nodeY(nullptr), d_nodeParent(nullptr), d_nodeTime(nullptr),
        h_nodeCount(0), d_nodeCapacity(0),
        d_obstacleX(nullptr), d_obstacleY(nullptr), d_obstacleWidth(nullptr), d_obstacleHeight(nullptr),
        h_obstacleCount(0), d_randStates(nullptr) {}
        
    // Destructor to free CUDA memory
    ~RRTCudaData();
};

// Function declarations
// Initialize CUDA resources
void initCudaRRT(RRTCudaData& data, int maxNodes, int numObstacles, int numThreads);

// Clean up CUDA resources
void cleanupCudaRRT(RRTCudaData& data);

// Find nearest node in the tree to the given point using CUDA
int findNearestCuda(RRTCudaData& data, float x, float y);

// Check if path between two nodes collides with any obstacle using CUDA
bool checkCollisionCuda(RRTCudaData& data, float x1, float y1, float x2, float y2);

// Generate random node in the configuration space using CUDA
void generateRandomNodeCuda(RRTCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax, 
                          float goalBias, float goalX, float goalY);

// Add a new node to the tree
void addNodeCuda(RRTCudaData& data, float x, float y, int parent, float time);

// Main CUDA RRT algorithm
std::vector<Node> buildRRTCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize = 0.1,
    double goalThreshold = 0.1,
    int maxIterations = 1000,
    double xMin = 0.0,
    double xMax = 1.0,
    double yMin = 0.0,
    double yMax = 1.0,
    const std::string& treeFilename = "rrt_cuda_tree.csv",
    bool enableVisualization = true,
    int numThreads = 256
);

// Helper functions
std::vector<Node> extractPathCuda(const RRTCudaData& data, int goalIndex);
void saveTreeToFileCuda(const RRTCudaData& data, const std::string& filename);

#endif // RRT_CUDA_H