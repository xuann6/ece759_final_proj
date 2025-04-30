// rrtBidirectionalCuda.h
#ifndef RRT_BIDIRECTIONAL_CUDA_H
#define RRT_BIDIRECTIONAL_CUDA_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "rrtCommon.h" // shared Node/Obstacle definitions

// Helper structure for CUDA device data
struct RRTBiCudaData {
    // Host vectors
    std::vector<Node> h_nodes;
    std::vector<Obstacle> h_obstacles;
    
    // Device arrays
    float* d_nodeX;       // Node x coordinates
    float* d_nodeY;       // Node y coordinates
    int* d_nodeParent;    // Node parent indices
    float* d_nodeTime;    // Node timestamps
    int* d_nodeTree;      // Which tree the node belongs to
    int h_nodeCount;      // Current number of nodes
    int d_nodeCapacity;   // Allocated capacity for nodes
    
    // Tree statistics
    int h_startTreeSize;  // Number of nodes in the start tree
    int h_goalTreeSize;   // Number of nodes in the goal tree
    
    // Connection information
    bool h_treesConnected;
    int h_startConnectIdx;
    int h_goalConnectIdx;
    
    // Obstacle data on device
    float* d_obstacleX;
    float* d_obstacleY;
    float* d_obstacleWidth;
    float* d_obstacleHeight;
    int h_obstacleCount;
    
    // Random state for CUDA
    curandState* d_randStates;
    
    // Constructor
    RRTBiCudaData() : 
        d_nodeX(nullptr), d_nodeY(nullptr), d_nodeParent(nullptr), 
        d_nodeTime(nullptr), d_nodeTree(nullptr),
        h_nodeCount(0), d_nodeCapacity(0),
        h_startTreeSize(0), h_goalTreeSize(0),
        h_treesConnected(false), h_startConnectIdx(-1), h_goalConnectIdx(-1),
        d_obstacleX(nullptr), d_obstacleY(nullptr), 
        d_obstacleWidth(nullptr), d_obstacleHeight(nullptr),
        h_obstacleCount(0), d_randStates(nullptr) {}
        
    // Destructor to free CUDA memory
    ~RRTBiCudaData();
};

// Function declarations
// Initialize CUDA resources
void initCudaRRTBi(RRTBiCudaData& data, int maxNodes, int numObstacles, int numThreads);

// Clean up CUDA resources
void cleanupCudaRRTBi(RRTBiCudaData& data);

// Find nearest node in the specified tree to the given point using CUDA
int findNearestInTreeCuda(RRTBiCudaData& data, float x, float y, int treeIdx);

// Find the closest pair of nodes between the two trees
std::pair<int, int> findClosestNodesBetweenTreesCuda(RRTBiCudaData& data);

// Check if path between two nodes collides with any obstacle using CUDA
bool checkCollisionCuda(RRTBiCudaData& data, float x1, float y1, float x2, float y2);

// Check if trees can be connected (nodes are close enough and path is collision-free)
bool canConnectTreesCuda(RRTBiCudaData& data, int startIdx, int goalIdx, float threshold);

// Generate random node in the configuration space using CUDA
void generateRandomNodeCuda(RRTBiCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax);

// Add a new node to the specified tree
void addNodeCuda(RRTBiCudaData& data, float x, float y, int parent, float time, int tree);

// Main CUDA Bidirectional RRT algorithm
std::vector<Node> buildRRTBidirectionalCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize = 0.1,
    double connectThreshold = 0.1,
    int maxIterations = 1000,
    double xMin = 0.0,
    double xMax = 1.0,
    double yMin = 0.0,
    double yMax = 1.0,
    const std::string& treeFilename = "rrt_bi_cuda_tree.csv",
    bool enableVisualization = true,
    int numThreads = 256
);

// Helper functions
std::vector<Node> extractBidirectionalPathCuda(RRTBiCudaData& data);
void saveTreeToFileCuda(const RRTBiCudaData& data, const std::string& filename);

#endif // RRT_BIDIRECTIONAL_CUDA_H