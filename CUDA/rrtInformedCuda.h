#ifndef RRT_INFORMED_CUDA_H
#define RRT_INFORMED_CUDA_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include "rrtCommon.h" // shared Node/Obstacle definitions

// Structure to represent an ellipsoidal sampling domain
struct EllipsoidSamplingDomain {
    float centerX, centerY;  // Center of the ellipsoid
    float transverse_axis;   // Length of the major axis (cbest)
    float conjugate_axis;    // Length of the minor axis (sqrt(cbest^2 - cmin^2))
    float cmin;              // Minimum possible cost (Euclidean distance between start and goal)
    
    // Pre-computed rotation matrix for the ellipsoid
    float rotMatrix[4];      // 2x2 rotation matrix stored as [a11, a12, a21, a22]
    
    EllipsoidSamplingDomain() 
        : centerX(0.0f), centerY(0.0f), transverse_axis(0.0f), conjugate_axis(0.0f), cmin(0.0f) {
        rotMatrix[0] = 1.0f; rotMatrix[1] = 0.0f;
        rotMatrix[2] = 0.0f; rotMatrix[3] = 1.0f;
    }
        
    EllipsoidSamplingDomain(float startX, float startY, float goalX, float goalY, float current_best_cost) 
        : centerX((startX + goalX) / 2.0f),
          centerY((startY + goalY) / 2.0f),
          transverse_axis(current_best_cost) {
        // Calculate minimum cost (Euclidean distance from start to goal)
        float dx = goalX - startX;
        float dy = goalY - startY;
        cmin = sqrtf(dx*dx + dy*dy);
        
        // Calculate conjugate axis length
        if (current_best_cost > cmin) {
            conjugate_axis = sqrtf(current_best_cost * current_best_cost - cmin * cmin);
        } else {
            conjugate_axis = 0.0f;
        }
        
        // Pre-compute rotation matrix
        if (cmin > 0.0001f) {
            // Direction vector from start to goal (normalized)
            float a1_x = dx / cmin;
            float a1_y = dy / cmin;
            
            // Perpendicular direction
            float a2_x = -a1_y;
            float a2_y = a1_x;
            
            // Store rotation matrix
            rotMatrix[0] = a1_x; rotMatrix[1] = a2_x;
            rotMatrix[2] = a1_y; rotMatrix[3] = a2_y;
        }
    }
    
    // Check if the domain is valid
    bool isValid() const {
        return (transverse_axis > cmin) && (cmin > 0) && (conjugate_axis > 0);
    }
};

// Helper structure for CUDA device data
struct RRTInformedCudaData {
    // Host vectors
    std::vector<Node> h_nodes;
    std::vector<Obstacle> h_obstacles;
    
    // Device arrays
    float* d_nodeX;       // Node x coordinates
    float* d_nodeY;       // Node y coordinates
    int* d_nodeParent;    // Node parent indices
    float* d_nodeTime;    // Node timestamps
    float* d_nodeCost;    // Node costs from start
    int h_nodeCount;      // Current number of nodes
    int d_nodeCapacity;   // Allocated capacity for nodes
    
    // Best solution found so far
    float h_bestCost;
    int h_goalNodeIndex;
    
    // Sampling domain data
    float h_startX, h_startY;
    float h_goalX, h_goalY;
    
    // Obstacle data on device
    float* d_obstacleX;
    float* d_obstacleY;
    float* d_obstacleWidth;
    float* d_obstacleHeight;
    int h_obstacleCount;
    
    // For optimized collision detection
    bool* d_obstacleGrid;     // Grid-based representation of obstacles
    int h_gridSize;           // Size of the spatial grid
    
    // World bounds
    float h_worldXMin, h_worldXMax;
    float h_worldYMin, h_worldYMax;
    
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
    
    // Batch sampling data
    float* d_batchSamplesX;   // X coordinates for batch samples
    float* d_batchSamplesY;   // Y coordinates for batch samples
    int* d_sampleResults;     // Results of sample processing
    int h_batchSize;          // Size of sample batch
    
    // Ellipsoid parameters for GPU
    float* d_ellipsoidParams; // [centerX, centerY, r1, r2, rotMatrix[4]]
    
    // Constructor
    RRTInformedCudaData() : 
        d_nodeX(nullptr), d_nodeY(nullptr), d_nodeParent(nullptr), 
        d_nodeTime(nullptr), d_nodeCost(nullptr),
        h_nodeCount(0), d_nodeCapacity(0),
        h_bestCost(FLT_MAX), h_goalNodeIndex(-1),
        h_startX(0.0f), h_startY(0.0f), h_goalX(0.0f), h_goalY(0.0f),
        d_obstacleX(nullptr), d_obstacleY(nullptr), 
        d_obstacleWidth(nullptr), d_obstacleHeight(nullptr),
        h_obstacleCount(0), d_obstacleGrid(nullptr), h_gridSize(0),
        h_worldXMin(0.0f), h_worldXMax(1.0f),
        h_worldYMin(0.0f), h_worldYMax(1.0f),
        d_randStates(nullptr),
        d_tempX(nullptr), d_tempY(nullptr), d_distances(nullptr),
        d_collisionResult(nullptr), d_inRadius(nullptr),
        d_bestParent(nullptr), d_minCost(nullptr),
        d_batchSamplesX(nullptr), d_batchSamplesY(nullptr),
        d_sampleResults(nullptr), h_batchSize(0),
        d_ellipsoidParams(nullptr) {}
        
    // Destructor to free CUDA memory
    ~RRTInformedCudaData();
};

// Function declarations
// Initialize CUDA resources
void initCudaRRTInformed(RRTInformedCudaData& data, int maxNodes, int numObstacles, int numThreads);

// Initialize ellipsoid parameters for GPU
void updateEllipsoidParams(RRTInformedCudaData& data);

// Initialize spatial grid for optimized collision detection
void initObstacleGrid(RRTInformedCudaData& data, int gridSize = 64);

// Clean up CUDA resources
void cleanupCudaRRTInformed(RRTInformedCudaData& data);

// Find nearest node in the tree to the given point using CUDA
int findNearestCuda(RRTInformedCudaData& data, float x, float y);

// Find all nodes within a radius using CUDA
std::vector<int> findNodesInRadiusCuda(RRTInformedCudaData& data, float x, float y, float radius);

// Choose best parent from neighbors using CUDA
int chooseBestParentCuda(RRTInformedCudaData& data, float x, float y, 
                         const std::vector<int>& neighbors);

// Check if path between two nodes collides with any obstacle using CUDA
bool checkCollisionCuda(RRTInformedCudaData& data, float x1, float y1, float x2, float y2);

// Rewire the tree using CUDA
void rewireTreeCuda(RRTInformedCudaData& data, int newNodeIdx, const std::vector<int>& neighbors);

// Sample a point from the informed subset (ellipsoid) using CUDA
void sampleInformedSubsetCuda(RRTInformedCudaData& data, float& x, float& y, 
                             float xMin, float xMax, float yMin, float yMax, float goalBias);

// Generate batch of samples from informed subset
void generateSampleBatchCuda(RRTInformedCudaData& data, int batchSize, 
                           float xMin, float xMax, float yMin, float yMax, float goalBias);

// Process batch of samples (find valid ones)
std::vector<std::pair<float, float>> processSampleBatchCuda(RRTInformedCudaData& data, 
                                                         float stepSize, 
                                                         const std::vector<Obstacle>& obstacles);

// Add a new node to the tree
void addNodeCuda(RRTInformedCudaData& data, float x, float y, int parent, float time, float cost);

// Update best cost when a better solution is found
void updateBestCostCuda(RRTInformedCudaData& data, float newCost, int goalNodeIndex);

// Update costs of descendants after rewiring
void updateDescendantCostsCuda(RRTInformedCudaData& data, int nodeIdx);

// Main CUDA Informed RRT* algorithm
std::vector<Node> buildRRTInformedCuda(
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
    const std::string& treeFilename = "rrt_informed_cuda_tree.csv",
    bool enableVisualization = true,
    int numThreads = 256,
    bool stopAtFirstSolution = false
);

// Helper functions
std::vector<Node> extractPathCuda(const RRTInformedCudaData& data, int goalIndex);
void saveTreeToFileCuda(const RRTInformedCudaData& data, const std::string& filename);

#endif // RRT_INFORMED_CUDA_H