// rrtBidirectionalCuda.cu
#include "rrtBidirectionalCuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

// Define block size for CUDA kernels
#define BLOCK_SIZE 256

// CUDA error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Destructor implementation
RRTBiCudaData::~RRTBiCudaData() {
    // Free device memory
    if (d_nodeX) cudaFree(d_nodeX);
    if (d_nodeY) cudaFree(d_nodeY);
    if (d_nodeParent) cudaFree(d_nodeParent);
    if (d_nodeTime) cudaFree(d_nodeTime);
    if (d_nodeTree) cudaFree(d_nodeTree);
    
    if (d_obstacleX) cudaFree(d_obstacleX);
    if (d_obstacleY) cudaFree(d_obstacleY);
    if (d_obstacleWidth) cudaFree(d_obstacleWidth);
    if (d_obstacleHeight) cudaFree(d_obstacleHeight);
    
    if (d_randStates) cudaFree(d_randStates);
}

// CUDA kernel to initialize random states
__global__ void initRandStatesKernel(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel to find the nearest node to a query point in the specified tree
__global__ void findNearestInTreeKernel(float* nodeX, float* nodeY, int* nodeTree,
                                       int nodeCount, float queryX, float queryY, 
                                       int treeIdx, float* distances, int* validNode) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        // Only consider nodes in the specified tree
        if (nodeTree[idx] == treeIdx) {
            float dx = nodeX[idx] - queryX;
            float dy = nodeY[idx] - queryY;
            distances[idx] = dx*dx + dy*dy; // Squared distance (faster than sqrt)
            validNode[idx] = 1;
        } else {
            distances[idx] = FLT_MAX;
            validNode[idx] = 0;
        }
    }
}

// CUDA kernel to find the closest pairs between two trees
__global__ void findClosestPairsKernel(float* nodeX, float* nodeY, int* nodeTree,
                                      int nodeCount, float* distanceMatrix, 
                                      int startTreeSize, int goalTreeSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate row and column from linear index
    int startIdx = idx / goalTreeSize;
    int goalIdx = idx % goalTreeSize;
    
    if (startIdx < startTreeSize && goalIdx < goalTreeSize) {
        // Find the actual indices in the node arrays
        int actualStartIdx = -1;
        int actualGoalIdx = -1;
        int startCount = 0;
        int goalCount = 0;
        
        for (int i = 0; i < nodeCount; i++) {
            if (nodeTree[i] == 0) {
                if (startCount == startIdx) {
                    actualStartIdx = i;
                }
                startCount++;
            } else if (nodeTree[i] == 1) {
                if (goalCount == goalIdx) {
                    actualGoalIdx = i;
                }
                goalCount++;
            }
            
            // Break early if both indices found
            if (actualStartIdx != -1 && actualGoalIdx != -1) {
                break;
            }
        }
        
        if (actualStartIdx != -1 && actualGoalIdx != -1) {
            float dx = nodeX[actualStartIdx] - nodeX[actualGoalIdx];
            float dy = nodeY[actualStartIdx] - nodeY[actualGoalIdx];
            float distSq = dx*dx + dy*dy;
            
            // Store distance and indices
            int linearIdx = startIdx * goalTreeSize + goalIdx;
            distanceMatrix[linearIdx] = distSq;
        }
    }
}

// CUDA kernel to check collision with obstacles
__global__ void checkCollisionKernel(float x1, float y1, float x2, float y2,
                                   float* obstacleX, float* obstacleY,
                                   float* obstacleWidth, float* obstacleHeight,
                                   int obstacleCount, bool* collisionResult) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Shared flag for collision detection
    __shared__ bool sharedCollision;
    
    // Initialize shared flag in the first thread
    if (threadIdx.x == 0) {
        sharedCollision = false;
    }
    __syncthreads();
    
    if (idx < obstacleCount && !sharedCollision) {
        // Line segment parameters
        float dx = x2 - x1;
        float dy = y2 - y1;
        float lineLength = sqrtf(dx*dx + dy*dy);
        
        // Check a number of points along the line segment
        const int NUM_CHECKS = 10;
        bool localCollision = false;
        
        for (int i = 0; i <= NUM_CHECKS && !localCollision; i++) {
            float t = (float)i / NUM_CHECKS;
            float x = x1 + t * dx;
            float y = y1 + t * dy;
            
            // Check if point is inside the obstacle
            if (x >= obstacleX[idx] && x <= obstacleX[idx] + obstacleWidth[idx] &&
                y >= obstacleY[idx] && y <= obstacleY[idx] + obstacleHeight[idx]) {
                localCollision = true;
            }
        }
        
        // If collision detected, set the shared flag
        if (localCollision) {
            atomicExch((int*)&sharedCollision, true);
        }
    }
    
    __syncthreads();
    
    // Only one thread updates the final result
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *collisionResult = sharedCollision;
    }
}

// CUDA kernel for random sampling in configuration space
__global__ void generateRandomNodeKernel(curandState* randStates, float* x, float* y,
                                      float xMin, float xMax, float yMin, float yMax) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Generate random position in configuration space
    *x = xMin + curand_uniform(&randStates[idx]) * (xMax - xMin);
    *y = yMin + curand_uniform(&randStates[idx]) * (yMax - yMin);
}

// Function to initialize CUDA resources
void initCudaRRTBi(RRTBiCudaData& data, int maxNodes, int numObstacles, int numThreads) {
    // Allocate memory for nodes
    data.d_nodeCapacity = maxNodes;
    CUDA_CHECK(cudaMalloc(&data.d_nodeX, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeY, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeParent, maxNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeTime, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeTree, maxNodes * sizeof(int)));
    
    // Allocate memory for obstacles
    if (numObstacles > 0) {
        CUDA_CHECK(cudaMalloc(&data.d_obstacleX, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleY, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleWidth, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleHeight, numObstacles * sizeof(float)));
    }
    
    // Initialize random states
    CUDA_CHECK(cudaMalloc(&data.d_randStates, numThreads * sizeof(curandState)));
    
    int blocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandStatesKernel<<<blocks, BLOCK_SIZE>>>(data.d_randStates, 
                                               static_cast<unsigned long>(time(nullptr)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize counters
    data.h_nodeCount = 0;
    data.h_startTreeSize = 0;
    data.h_goalTreeSize = 0;
    data.h_treesConnected = false;
    data.h_startConnectIdx = -1;
    data.h_goalConnectIdx = -1;
}

// Function to clean up CUDA resources
void cleanupCudaRRTBi(RRTBiCudaData& data) {
    // Most cleanup is handled by the destructor
    // This function is provided for explicit cleanup if needed
    cudaDeviceSynchronize();
}

// Function to find nearest node in the specified tree using CUDA
int findNearestInTreeCuda(RRTBiCudaData& data, float x, float y, int treeIdx) {
    if (data.h_nodeCount == 0) {
        return -1;
    }
    
    // Allocate memory for distances and valid node flags
    float* d_distances;
    int* d_validNode;
    CUDA_CHECK(cudaMalloc(&d_distances, data.h_nodeCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_validNode, data.h_nodeCount * sizeof(int)));
    
    // Calculate number of blocks needed
    int blocks = (data.h_nodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to compute distances
    findNearestInTreeKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.d_nodeTree, data.h_nodeCount, 
        x, y, treeIdx, d_distances, d_validNode);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Find minimum distance index using Thrust
    thrust::device_ptr<float> thrust_distances(d_distances);
    thrust::device_ptr<int> thrust_validNode(d_validNode);
    
    // Use a large initial value
    float minDist = FLT_MAX;
    int minIdx = -1;
    
    // Copy back to host for processing (more efficient for small trees)
    std::vector<float> h_distances(data.h_nodeCount);
    std::vector<int> h_validNode(data.h_nodeCount);
    
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, 
                        data.h_nodeCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_validNode.data(), d_validNode, 
                        data.h_nodeCount * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Find minimum distance among valid nodes
    for (int i = 0; i < data.h_nodeCount; i++) {
        if (h_validNode[i] && h_distances[i] < minDist) {
            minDist = h_distances[i];
            minIdx = i;
        }
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_validNode));
    
    return minIdx;
}

// Function to find the closest pair of nodes between the two trees
std::pair<int, int> findClosestNodesBetweenTreesCuda(RRTBiCudaData& data) {
    if (data.h_startTreeSize == 0 || data.h_goalTreeSize == 0) {
        return std::make_pair(-1, -1);
    }
    
    // Approach 1: For small trees, we can use a brute force approach on the host
    if (data.h_startTreeSize * data.h_goalTreeSize < 10000) {
        int bestStartIdx = -1;
        int bestGoalIdx = -1;
        float minDistSq = FLT_MAX;
        
        for (int i = 0; i < data.h_nodeCount; i++) {
            if (data.h_nodes[i].tree == 0) { // Start tree
                for (int j = 0; j < data.h_nodeCount; j++) {
                    if (data.h_nodes[j].tree == 1) { // Goal tree
                        float dx = data.h_nodes[i].x - data.h_nodes[j].x;
                        float dy = data.h_nodes[i].y - data.h_nodes[j].y;
                        float distSq = dx*dx + dy*dy;
                        
                        if (distSq < minDistSq) {
                            minDistSq = distSq;
                            bestStartIdx = i;
                            bestGoalIdx = j;
                        }
                    }
                }
            }
        }
        
        return std::make_pair(bestStartIdx, bestGoalIdx);
    }
    
    // Approach 2: For larger trees, use CUDA to compute all pairs of distances
    // Allocate memory for the distance matrix
    float* d_distanceMatrix;
    CUDA_CHECK(cudaMalloc(&d_distanceMatrix, 
                        data.h_startTreeSize * data.h_goalTreeSize * sizeof(float)));
    
    // Initialize to large values
    CUDA_CHECK(cudaMemset(d_distanceMatrix, 0xFF, 
                        data.h_startTreeSize * data.h_goalTreeSize * sizeof(float)));
    
    // Launch kernel to compute distances between trees
    int totalPairs = data.h_startTreeSize * data.h_goalTreeSize;
    int blocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    findClosestPairsKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.d_nodeTree, data.h_nodeCount, 
        d_distanceMatrix, data.h_startTreeSize, data.h_goalTreeSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Find minimum distance index using Thrust
    thrust::device_ptr<float> thrust_distances(d_distanceMatrix);
    thrust::device_ptr<float> min_distance_ptr = thrust::min_element(
        thrust_distances, thrust_distances + totalPairs);
    
    // Get the linear index of the minimum element
    int minLinearIdx = min_distance_ptr - thrust_distances;
    
    // Convert linear index to start and goal indices
    int startMatrixIdx = minLinearIdx / data.h_goalTreeSize;
    int goalMatrixIdx = minLinearIdx % data.h_goalTreeSize;
    
    // Now convert matrix indices to actual node indices
    int actualStartIdx = -1;
    int actualGoalIdx = -1;
    int startCount = 0;
    int goalCount = 0;
    
    for (int i = 0; i < data.h_nodeCount; i++) {
        if (data.h_nodes[i].tree == 0) {
            if (startCount == startMatrixIdx) {
                actualStartIdx = i;
            }
            startCount++;
        } else if (data.h_nodes[i].tree == 1) {
            if (goalCount == goalMatrixIdx) {
                actualGoalIdx = i;
            }
            goalCount++;
        }
        
        // Break early if both indices found
        if (actualStartIdx != -1 && actualGoalIdx != -1) {
            break;
        }
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_distanceMatrix));
    
    return std::make_pair(actualStartIdx, actualGoalIdx);
}

// Function to check collision using CUDA
bool checkCollisionCuda(RRTBiCudaData& data, float x1, float y1, float x2, float y2) {
    if (data.h_obstacleCount == 0) {
        return false; // No obstacles, no collision
    }
    
    // Allocate memory for collision result
    bool* d_collisionResult;
    CUDA_CHECK(cudaMalloc(&d_collisionResult, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_collisionResult, 0, sizeof(bool)));
    
    // Calculate number of blocks needed
    int blocks = (data.h_obstacleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to check collisions
    checkCollisionKernel<<<blocks, BLOCK_SIZE>>>(
        x1, y1, x2, y2,
        data.d_obstacleX, data.d_obstacleY, data.d_obstacleWidth, data.d_obstacleHeight,
        data.h_obstacleCount, d_collisionResult);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get the result
    bool collisionResult;
    CUDA_CHECK(cudaMemcpy(&collisionResult, d_collisionResult, sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Free memory
    CUDA_CHECK(cudaFree(d_collisionResult));
    
    return collisionResult;
}

// Function to check if trees can be connected
bool canConnectTreesCuda(RRTBiCudaData& data, int startIdx, int goalIdx, float threshold) {
    if (startIdx < 0 || goalIdx < 0 || startIdx >= data.h_nodeCount || goalIdx >= data.h_nodeCount) {
        return false;
    }
    
    // Get node coordinates
    float startX = data.h_nodes[startIdx].x;
    float startY = data.h_nodes[startIdx].y;
    float goalX = data.h_nodes[goalIdx].x;
    float goalY = data.h_nodes[goalIdx].y;
    
    // Check distance
    float dx = startX - goalX;
    float dy = startY - goalY;
    float distSq = dx*dx + dy*dy;
    
    if (distSq > threshold*threshold) {
        return false;
    }
    
    // Check for collision
    return !checkCollisionCuda(data, startX, startY, goalX, goalY);
}

// Function to generate random node using CUDA
void generateRandomNodeCuda(RRTBiCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax) {
    // Allocate memory for result
    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
    
    // Launch kernel to generate random node
    // Using just one thread since we need only one random node
    generateRandomNodeKernel<<<1, 1>>>(data.d_randStates, d_x, d_y, 
                                     xMin, xMax, yMin, yMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(&x, d_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free memory
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// Steer function (implemented on host for simplicity)
void steerCuda(float x1, float y1, float x2, float y2, float stepSize, float& newX, float& newY) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dist = sqrtf(dx*dx + dy*dy);
    
    if (dist <= stepSize) {
        newX = x2;
        newY = y2;
    } else {
        float ratio = stepSize / dist;
        newX = x1 + ratio * dx;
        newY = y1 + ratio * dy;
    }
}

// Function to add a new node to the tree
void addNodeCuda(RRTBiCudaData& data, float x, float y, int parent, float time, int tree) {
    if (data.h_nodeCount >= data.d_nodeCapacity) {
        std::cerr << "Error: Node capacity exceeded" << std::endl;
        return;
    }
    
    // Add to host vector for easier path extraction later
    data.h_nodes.push_back(Node(x, y, parent, time, tree));
    
    // Add to device arrays
    CUDA_CHECK(cudaMemcpy(&data.d_nodeX[data.h_nodeCount], &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeY[data.h_nodeCount], &y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[data.h_nodeCount], &parent, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeTime[data.h_nodeCount], &time, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeTree[data.h_nodeCount], &tree, sizeof(int), cudaMemcpyHostToDevice));
    
    // Update tree size counters
    if (tree == 0) {
        data.h_startTreeSize++;
    } else {
        data.h_goalTreeSize++;
    }
    
    data.h_nodeCount++;
}

// Extract bidirectional path from start to goal
std::vector<Node> extractBidirectionalPathCuda(RRTBiCudaData& data) {
    std::vector<Node> path;
    
    if (!data.h_treesConnected) {
        return path; // Empty path if trees not connected
    }
    
    // Extract path from start tree
    int currentIndex = data.h_startConnectIdx;
    std::vector<Node> startPath;
    
    while (currentIndex != -1) {
        startPath.push_back(data.h_nodes[currentIndex]);
        currentIndex = data.h_nodes[currentIndex].parent;
    }
    
    // Reverse path from start (to get path from start to connection point)
    std::reverse(startPath.begin(), startPath.end());
    
    // Add start path to result
    path.insert(path.end(), startPath.begin(), startPath.end());
    
    // Extract path from goal tree (from connection point to goal)
    currentIndex = data.h_goalConnectIdx;
    
    // Skip the first node if it's duplicating the connection point
    bool isFirst = true;
    
    while (currentIndex != -1) {
        if (!isFirst) {
            // Add node to path (from goal tree)
            path.push_back(data.h_nodes[currentIndex]);
        } else {
            isFirst = false;
        }
        currentIndex = data.h_nodes[currentIndex].parent;
    }
    
    // Reverse the goal path to get proper direction (from connection to goal)
    size_t startPathSize = startPath.size();
    std::reverse(path.begin() + startPathSize, path.end());
    
    return path;
}

// Save tree data to file for visualization
void saveTreeToFileCuda(const RRTBiCudaData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "node_id,x,y,parent_id,time,tree" << std::endl;
    
    // Write node data
    for (int i = 0; i < data.h_nodeCount; i++) {
        file << i << ","
             << data.h_nodes[i].x << ","
             << data.h_nodes[i].y << ","
             << data.h_nodes[i].parent << ","
             << data.h_nodes[i].time << ","
             << data.h_nodes[i].tree << std::endl;
    }
    
    file.close();
    std::cout << "Tree data saved to " << filename << std::endl;
}

// Main CUDA RRT Bidirectional algorithm
std::vector<Node> buildRRTBidirectionalCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize,
    double connectThreshold,
    int maxIterations,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    const std::string& treeFilename,
    bool enableVisualization,
    int numThreads
) {
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize CUDA data
    RRTBiCudaData data;
    initCudaRRTBi(data, maxIterations + 2, obstacles.size(), numThreads); // +2 for start and goal
    
    // Add start node to start tree (tree = 0)
    addNodeCuda(data, start.x, start.y, -1, 0.0, 0);
    
    // Add goal node to goal tree (tree = 1)
    addNodeCuda(data, goal.x, goal.y, -1, 0.0, 1);
    
    // Copy obstacles to device
    data.h_obstacleCount = obstacles.size();
    if (obstacles.size() > 0) {
        std::vector<float> h_obstacleX(obstacles.size());
        std::vector<float> h_obstacleY(obstacles.size());
        std::vector<float> h_obstacleWidth(obstacles.size());
        std::vector<float> h_obstacleHeight(obstacles.size());
        
        for (size_t i = 0; i < obstacles.size(); i++) {
            h_obstacleX[i] = obstacles[i].x;
            h_obstacleY[i] = obstacles[i].y;
            h_obstacleWidth[i] = obstacles[i].width;
            h_obstacleHeight[i] = obstacles[i].height;
        }
        
        CUDA_CHECK(cudaMemcpy(data.d_obstacleX, h_obstacleX.data(), obstacles.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data.d_obstacleY, h_obstacleY.data(), obstacles.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data.d_obstacleWidth, h_obstacleWidth.data(), obstacles.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data.d_obstacleHeight, h_obstacleHeight.data(), obstacles.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Flag to alternate between trees
    bool expandStartTree = true;
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Get current time for visualization
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        float timeSeconds = static_cast<float>(elapsed.count());
        
        // Determine which tree to expand
        int currentTree = expandStartTree ? 0 : 1;
        int otherTree = expandStartTree ? 1 : 0;
        
        // Generate random node
        float randomX, randomY;
        generateRandomNodeCuda(data, randomX, randomY, xMin, xMax, yMin, yMax);
        
        // Skip if inside obstacle
        bool insideObstacle = false;
        for (const auto& obstacle : obstacles) {
            if (randomX >= obstacle.x && randomX <= obstacle.x + obstacle.width &&
                randomY >= obstacle.y && randomY <= obstacle.y + obstacle.height) {
                insideObstacle = true;
                break;
            }
        }
        
        if (insideObstacle) {
            continue;
        }
        
        // Find nearest node in the current tree
        int nearestIndex = findNearestInTreeCuda(data, randomX, randomY, currentTree);
        
        // Skip if no nearest node found (should not happen)
        if (nearestIndex < 0) {
            continue;
        }
        
        // Steer towards random node
        float newX, newY;
        steerCuda(data.h_nodes[nearestIndex].x, data.h_nodes[nearestIndex].y, 
                 randomX, randomY, stepSize, newX, newY);
        
        // Check for collision
        if (checkCollisionCuda(data, data.h_nodes[nearestIndex].x, data.h_nodes[nearestIndex].y, 
                              newX, newY)) {
            continue; // Skip if collision
        }
        
        // Add new node to current tree
        addNodeCuda(data, newX, newY, nearestIndex, timeSeconds, currentTree);
        int newNodeIndex = data.h_nodeCount - 1;
        
        // Try to connect to other tree
        int nearestInOtherTree = findNearestInTreeCuda(data, newX, newY, otherTree);
        
        if (nearestInOtherTree >= 0) {
            if (canConnectTreesCuda(data, newNodeIndex, nearestInOtherTree, connectThreshold)) {
                // Trees connected!
                data.h_treesConnected = true;
                
                // Store connection indices
                if (currentTree == 0) {
                    data.h_startConnectIdx = newNodeIndex;
                    data.h_goalConnectIdx = nearestInOtherTree;
                } else {
                    data.h_startConnectIdx = nearestInOtherTree;
                    data.h_goalConnectIdx = newNodeIndex;
                }
                
                std::cout << "Trees connected at iteration " << i << std::endl;
                
                // Save tree data if visualization is enabled
                if (enableVisualization) {
                    saveTreeToFileCuda(data, treeFilename);
                }
                
                // Extract and return path
                std::vector<Node> path = extractBidirectionalPathCuda(data);
                
                // Clean up CUDA resources
                cleanupCudaRRTBi(data);
                
                return path;
            }
        }
        
        // Alternate between trees
        expandStartTree = !expandStartTree;
        
        // Periodically check if trees can be connected directly
        if (i % 10 == 0) {
            auto closestPair = findClosestNodesBetweenTreesCuda(data);
            
            if (closestPair.first >= 0 && closestPair.second >= 0) {
                if (canConnectTreesCuda(data, closestPair.first, closestPair.second, connectThreshold)) {
                    // Trees connected!
                    data.h_treesConnected = true;
                    data.h_startConnectIdx = closestPair.first;
                    data.h_goalConnectIdx = closestPair.second;
                    
                    std::cout << "Trees connected at iteration " << i << " (direct check)" << std::endl;
                    
                    // Save tree data if visualization is enabled
                    if (enableVisualization) {
                        saveTreeToFileCuda(data, treeFilename);
                    }
                    
                    // Extract and return path
                    std::vector<Node> path = extractBidirectionalPathCuda(data);
                    
                    // Clean up CUDA resources
                    cleanupCudaRRTBi(data);
                    
                    return path;
                }
            }
        }
        
        // Periodic visualization
        if (enableVisualization && i % 100 == 0) {
            saveTreeToFileCuda(data, treeFilename);
        }
    }
    
    // If max iterations reached without connecting trees
    std::cout << "Max iterations reached without connecting trees." << std::endl;
    
    // Save final tree data if visualization is enabled
    if (enableVisualization) {
        saveTreeToFileCuda(data, treeFilename);
    }
    
    // Clean up CUDA resources
    cleanupCudaRRTBi(data);
    
    // Return empty path if trees not connected
    return std::vector<Node>();
}