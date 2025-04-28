#ifndef RRT_STAR_CUDA_H
#define RRT_STAR_CUDA_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "rrtCommon.h"  // Shared Node / Obstacle definitions

// Helper structure for CUDA device data
struct RRTStarCudaData {
    // Host vectors
    std::vector<Node> h_nodes;
    std::vector<Obstacle> h_obstacles;
    
    // Device arrays for nodes
    float* d_nodeX;       // Node x coordinates
    float* d_nodeY;       // Node y coordinates
    int* d_nodeParent;    // Node parent indices
    float* d_nodeTime;    // Node timestamps
    float* d_nodeCost;    // Node costs from start
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
    
    // Pre-allocated temporary buffers
    float* d_tempX;           // For random point X coordinates
    float* d_tempY;           // For random point Y coordinates
    float* d_distances;       // For distance calculations
    bool* d_collisionResult;  // For collision detection results
    int* d_inRadius;          // For radius check results
    int* d_bestParent;        // For best parent selection
    float* d_minCost;         // For minimum cost tracking
    
    // For optimized collision detection
    bool* d_obstacleGrid;     // Grid-based representation of obstacles
    int h_gridSize;           // Size of the spatial grid
    
    // World bounds
    float h_worldXMin, h_worldXMax;
    float h_worldYMin, h_worldYMax;
    
    // Constructor
    RRTStarCudaData() : 
        d_nodeX(nullptr), d_nodeY(nullptr), d_nodeParent(nullptr), 
        d_nodeTime(nullptr), d_nodeCost(nullptr),
        h_nodeCount(0), d_nodeCapacity(0),
        d_obstacleX(nullptr), d_obstacleY(nullptr), 
        d_obstacleWidth(nullptr), d_obstacleHeight(nullptr),
        h_obstacleCount(0), d_randStates(nullptr),
        d_tempX(nullptr), d_tempY(nullptr), d_distances(nullptr),
        d_collisionResult(nullptr), d_inRadius(nullptr),
        d_bestParent(nullptr), d_minCost(nullptr),
        d_obstacleGrid(nullptr), h_gridSize(0),
        h_worldXMin(0.0f), h_worldXMax(1.0f),
        h_worldYMin(0.0f), h_worldYMax(1.0f) {}
        
    // Destructor to free CUDA memory
    ~RRTStarCudaData();
};

// Function declarations
// Initialize CUDA resources
void initCudaRRTStar(RRTStarCudaData& data, int maxNodes, int numObstacles, int numThreads);

// Initialize spatial grid for optimized collision detection
void initObstacleGrid(RRTStarCudaData& data, int gridSize = 64);

// Clean up CUDA resources
void cleanupCudaRRTStar(RRTStarCudaData& data);

// Find nearest node in the tree to the given point using CUDA
int findNearestCuda(RRTStarCudaData& data, float x, float y);

// Find all nodes within a radius using CUDA
std::vector<int> findNodesInRadiusCuda(RRTStarCudaData& data, float x, float y, float radius);

// Choose best parent from neighbors using CUDA
int chooseBestParentCuda(RRTStarCudaData& data, float x, float y, 
                         const std::vector<int>& neighbors);

// Check if path between two nodes collides with any obstacle using CUDA
bool checkCollisionCuda(RRTStarCudaData& data, float x1, float y1, float x2, float y2);

// Rewire the tree using CUDA
void rewireTreeCuda(RRTStarCudaData& data, int newNodeIdx, const std::vector<int>& neighbors);

// Generate random node in the configuration space using CUDA
void generateRandomNodeCuda(RRTStarCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax, 
                          float goalBias, float goalX, float goalY);

// Add a new node to the tree
void addNodeCuda(RRTStarCudaData& data, float x, float y, int parent, float time, float cost);

// Update costs of descendants after rewiring
void updateDescendantCostsCuda(RRTStarCudaData& data, int nodeIdx);

// Main CUDA RRT* algorithm
std::vector<Node> buildRRTStarCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize = 0.1,
    double goalThreshold = 0.1,
    int maxIterations = 1000,
    double rewireRadius = 0.2,
    double xMin = 0.0,
    double xMax = 1.0,
    double yMin = 0.0,
    double yMax = 1.0,
    const std::string& treeFilename = "rrt_star_cuda_tree.csv",
    bool enableVisualization = true,
    int numThreads = 256
);

// Helper functions
std::vector<Node> extractPathCuda(const RRTStarCudaData& data, int goalIndex);
void saveTreeToFileCuda(const RRTStarCudaData& data, const std::string& filename);

#endif // RRT_STAR_CUDA_H