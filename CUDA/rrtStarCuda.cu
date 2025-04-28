// rrtStarCuda.cu
#include "rrtStarCuda.h"
#include "cudaRRTUtils.h"
#include "cudaRRTKernels.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

// Destructor implementation
RRTStarCudaData::~RRTStarCudaData() {
    // Free node data
    if (d_nodeX) cudaFree(d_nodeX);
    if (d_nodeY) cudaFree(d_nodeY);
    if (d_nodeParent) cudaFree(d_nodeParent);
    if (d_nodeTime) cudaFree(d_nodeTime);
    if (d_nodeCost) cudaFree(d_nodeCost);
    
    // Free obstacle data
    if (d_obstacleX) cudaFree(d_obstacleX);
    if (d_obstacleY) cudaFree(d_obstacleY);
    if (d_obstacleWidth) cudaFree(d_obstacleWidth);
    if (d_obstacleHeight) cudaFree(d_obstacleHeight);
    
    // Free temporary buffers
    if (d_tempX) cudaFree(d_tempX);
    if (d_tempY) cudaFree(d_tempY);
    if (d_distances) cudaFree(d_distances);
    if (d_collisionResult) cudaFree(d_collisionResult);
    if (d_inRadius) cudaFree(d_inRadius);
    if (d_bestParent) cudaFree(d_bestParent);
    if (d_minCost) cudaFree(d_minCost);
    
    // Free grid data
    if (d_obstacleGrid) cudaFree(d_obstacleGrid);
    
    // Free random states
    if (d_randStates) cudaFree(d_randStates);
}

// Kernel implementations are moved to cudaRRTKernels.h

// CUDA kernel to populate the obstacle grid
__global__ void populateObstacleGridKernel(float* obstacleX, float* obstacleY,
                                         float* obstacleWidth, float* obstacleHeight,
                                         int obstacleCount, bool* obstacleGrid, int gridSize,
                                         float xMin, float xMax, float yMin, float yMax) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < obstacleCount) {
        float ox = obstacleX[idx];
        float oy = obstacleY[idx];
        float width = obstacleWidth[idx];
        float height = obstacleHeight[idx];
        
        // Calculate grid cells covered by this obstacle
        int minCellX = (int)((ox - xMin) / (xMax - xMin) * gridSize);
        int minCellY = (int)((oy - yMin) / (yMax - yMin) * gridSize);
        int maxCellX = (int)(((ox + width) - xMin) / (xMax - xMin) * gridSize);
        int maxCellY = (int)(((oy + height) - yMin) / (yMax - yMin) * gridSize);
        
        // Clamp to grid bounds
        minCellX = max(0, min(gridSize-1, minCellX));
        minCellY = max(0, min(gridSize-1, minCellY));
        maxCellX = max(0, min(gridSize-1, maxCellX));
        maxCellY = max(0, min(gridSize-1, maxCellY));
        
        // Mark grid cells as occupied
        for (int y = minCellY; y <= maxCellY; y++) {
            for (int x = minCellX; x <= maxCellX; x++) {
                obstacleGrid[y * gridSize + x] = true;
            }
        }
    }
}

// CUDA kernel to check collision using grid
__global__ void checkCollisionGridKernel(float x1, float y1, float x2, float y2,
                                       bool* obstacleGrid, int gridSize,
                                       float xMin, float xMax, float yMin, float yMax,
                                       bool* collisionResult) {
    // Thread 0 traces the line through the grid
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Convert to grid coordinates
        int x1Cell = (int)((x1 - xMin) / (xMax - xMin) * gridSize);
        int y1Cell = (int)((y1 - yMin) / (yMax - yMin) * gridSize);
        int x2Cell = (int)((x2 - xMin) / (xMax - xMin) * gridSize);
        int y2Cell = (int)((y2 - yMin) / (yMax - yMin) * gridSize);
        
        // Clamp to grid bounds
        x1Cell = min(gridSize-1, max(0, x1Cell));
        y1Cell = min(gridSize-1, max(0, y1Cell));
        x2Cell = min(gridSize-1, max(0, x2Cell));
        y2Cell = min(gridSize-1, max(0, y2Cell));
        
        bool collision = false;
        
        // Bresenham line algorithm
        int dx = abs(x2Cell - x1Cell);
        int dy = abs(y2Cell - y1Cell);
        int sx = (x1Cell < x2Cell) ? 1 : -1;
        int sy = (y1Cell < y2Cell) ? 1 : -1;
        int err = dx - dy;
        
        int x = x1Cell;
        int y = y1Cell;
        
        while (true) {
            // Check grid cell for obstacle
            if (obstacleGrid[y * gridSize + x]) {
                collision = true;
                break;
            }
            
            // Check if reached end point
            if (x == x2Cell && y == y2Cell) break;
            
            // Bresenham step
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
        
        *collisionResult = collision;
    }
}

// Rewiring neighbors kernel is defined in cudaRRTKernels.h

// CUDA kernel to update costs of descendants after rewiring
__global__ void updateDescendantCostsKernel(float* nodeX, float* nodeY, 
                                         int* nodeParent, float* nodeCost,
                                         int nodeCount, int* changedNodes, 
                                         int changedCount, int* nextChangedNodes,
                                         int* nextChangedCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        // Check if any parent node has changed
        for (int i = 0; i < changedCount; i++) {
            int parentIdx = changedNodes[i];
            
            // If this is a child of the changed node
            if (nodeParent[idx] == parentIdx) {
                // Recalculate cost
                float dx = nodeX[idx] - nodeX[parentIdx];
                float dy = nodeY[idx] - nodeY[parentIdx];
                float edgeCost = sqrtf(dx*dx + dy*dy);
                float newCost = nodeCost[parentIdx] + edgeCost;
                
                // Update cost
                nodeCost[idx] = newCost;
                
                // Mark this node as changed for next wave of propagation
                int slot = atomicAdd(nextChangedCount, 1);
                if (slot < nodeCount) {  // Safety check
                    nextChangedNodes[slot] = idx;
                }
            }
        }
    }
}

// Additional kernel implementations moved to cudaRRTKernels.h

// Function to initialize CUDA resources
void initCudaRRTStar(RRTStarCudaData& data, int maxNodes, int numObstacles, int numThreads) {
    // Allocate memory for nodes
    data.d_nodeCapacity = maxNodes;
    CUDA_CHECK(cudaMalloc(&data.d_nodeX, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeY, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeParent, maxNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeTime, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeCost, maxNodes * sizeof(float)));
    
    // Allocate memory for obstacles
    if (numObstacles > 0) {
        CUDA_CHECK(cudaMalloc(&data.d_obstacleX, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleY, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleWidth, numObstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data.d_obstacleHeight, numObstacles * sizeof(float)));
    }
    
    // Allocate memory for temporary buffers
    CUDA_CHECK(cudaMalloc(&data.d_tempX, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_tempY, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_collisionResult, sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&data.d_bestParent, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_minCost, sizeof(float)));
    
    // Initialize random states
    CUDA_CHECK(cudaMalloc(&data.d_randStates, numThreads * sizeof(curandState)));
    
    int blocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandStatesKernel<<<blocks, BLOCK_SIZE>>>(data.d_randStates, 
                                               static_cast<unsigned long>(time(nullptr)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Initialize spatial grid for optimized collision detection
void initObstacleGrid(RRTStarCudaData& data, int gridSize) {
    // Set grid size
    data.h_gridSize = gridSize;
    
    // Allocate grid memory
    CUDA_CHECK(cudaMalloc(&data.d_obstacleGrid, gridSize * gridSize * sizeof(bool)));
    CUDA_CHECK(cudaMemset(data.d_obstacleGrid, 0, gridSize * gridSize * sizeof(bool)));
    
    // Only populate grid if there are obstacles
    if (data.h_obstacleCount > 0) {
        int blocks = (data.h_obstacleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        populateObstacleGridKernel<<<blocks, BLOCK_SIZE>>>(
            data.d_obstacleX, data.d_obstacleY, data.d_obstacleWidth, data.d_obstacleHeight,
            data.h_obstacleCount, data.d_obstacleGrid, gridSize,
            data.h_worldXMin, data.h_worldXMax, data.h_worldYMin, data.h_worldYMax);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Function to clean up CUDA resources
void cleanupCudaRRTStar(RRTStarCudaData& data) {
    // Handled by the destructor
    cudaDeviceSynchronize();
}

// Function to find nearest node using CUDA
int findNearestCuda(RRTStarCudaData& data, float x, float y) {
    if (data.h_nodeCount == 0) {
        return -1;
    }
    
    // Use pre-allocated buffers or allocate if needed
    if (data.d_distances == nullptr) {
        CUDA_CHECK(cudaMalloc(&data.d_distances, data.d_nodeCapacity * sizeof(float)));
    }
    
    // Calculate number of blocks needed
    int blocks = (data.h_nodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to compute distances
    findNearestKernel<<<blocks, BLOCK_SIZE>>>(data.d_nodeX, data.d_nodeY, data.h_nodeCount, 
                                          x, y, data.d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Find minimum distance index using Thrust
    thrust::device_ptr<float> thrust_distances(data.d_distances);
    thrust::device_ptr<float> min_distance_ptr = thrust::min_element(
        thrust_distances, thrust_distances + data.h_nodeCount);
    
    // Get the index of the minimum element
    int minIndex = static_cast<int>(min_distance_ptr - thrust_distances);
    
    return minIndex;
}

// Function to find nodes within a radius using CUDA
std::vector<int> findNodesInRadiusCuda(RRTStarCudaData& data, float x, float y, float radius) {
    std::vector<int> result;
    
    if (data.h_nodeCount == 0) {
        return result;
    }
    
    // Use pre-allocated buffers or allocate if needed
    if (data.d_inRadius == nullptr) {
        CUDA_CHECK(cudaMalloc(&data.d_inRadius, data.d_nodeCapacity * sizeof(int)));
    }
    
    // Reset the buffer
    CUDA_CHECK(cudaMemset(data.d_inRadius, 0, data.h_nodeCount * sizeof(int)));
    
    // Calculate number of blocks needed
    int blocks = (data.h_nodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to find nodes in radius
    float radiusSq = radius * radius;
    findNodesInRadiusKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.h_nodeCount,
        x, y, radiusSq, data.d_inRadius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<int> h_inRadius(data.h_nodeCount);
    CUDA_CHECK(cudaMemcpy(h_inRadius.data(), data.d_inRadius, 
                        data.h_nodeCount * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Build result vector
    for (int i = 0; i < data.h_nodeCount; i++) {
        if (h_inRadius[i] == 1) {
            result.push_back(i);
        }
    }
    
    return result;
}

// Function to choose best parent using CUDA
int chooseBestParentCuda(RRTStarCudaData& data, float x, float y, 
                       const std::vector<int>& neighbors) {
    if (neighbors.empty()) {
        return -1;
    }
    
    int neighborCount = neighbors.size();
    
    // Allocate device memory for neighbors and costs
    int* d_neighbors;
    float* d_costToNew;
    
    CUDA_CHECK(cudaMalloc(&d_neighbors, neighborCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_costToNew, neighborCount * sizeof(float)));
    
    // Copy neighbors to device
    CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors.data(), 
                        neighborCount * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize best parent to -1 and min cost to infinity
    int initialParent = -1;
    float initialCost = FLT_MAX;
    
    CUDA_CHECK(cudaMemcpy(data.d_bestParent, &initialParent, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_minCost, &initialCost, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel to find best parent
    int blocks = (neighborCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findBestParentKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.d_nodeCost,
        d_neighbors, neighborCount, x, y, d_costToNew, data.d_minCost, data.d_bestParent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get result
    int bestParent;
    CUDA_CHECK(cudaMemcpy(&bestParent, data.d_bestParent, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free memory
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_costToNew));
    
    return bestParent;
}

// Function to check collision using CUDA
bool checkCollisionCuda(RRTStarCudaData& data, float x1, float y1, float x2, float y2) {
    if (data.h_obstacleCount == 0) {
        return false; // No obstacles, no collision
    }
    
    // Reset collision result
    CUDA_CHECK(cudaMemset(data.d_collisionResult, 0, sizeof(bool)));
    
    // Use grid-based collision detection if grid is initialized
    if (data.h_gridSize > 0 && data.d_obstacleGrid != nullptr) {
        checkCollisionGridKernel<<<1, 1>>>(
            x1, y1, x2, y2,
            data.d_obstacleGrid, data.h_gridSize,
            data.h_worldXMin, data.h_worldXMax, data.h_worldYMin, data.h_worldYMax,
            data.d_collisionResult);
    } else {
        // Fall back to traditional collision detection
        // Calculate number of blocks needed
        int numBlocks = (data.h_obstacleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Launch kernel to check collisions
        checkCollisionKernel<<<numBlocks, BLOCK_SIZE>>>(
            x1, y1, x2, y2,
            data.d_obstacleX, data.d_obstacleY, data.d_obstacleWidth, data.d_obstacleHeight,
            data.h_obstacleCount, data.d_collisionResult);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get the result
    bool collisionResult;
    CUDA_CHECK(cudaMemcpy(&collisionResult, data.d_collisionResult, sizeof(bool), cudaMemcpyDeviceToHost));
    
    return collisionResult;
}

// Function to recursively update the costs of descendant nodes
void updateDescendantCostsCuda(RRTStarCudaData& data, int nodeIdx) {
    // Perform recursive cost update on the CPU (for simplicity)
    // Find all nodes that have this node as parent
    std::vector<int> children;
    
    for (int i = 0; i < data.h_nodeCount; i++) {
        if (data.h_nodes[i].parent == nodeIdx) {
            children.push_back(i);
            
            // Calculate new cost
            float dx = data.h_nodes[i].x - data.h_nodes[nodeIdx].x;
            float dy = data.h_nodes[i].y - data.h_nodes[nodeIdx].y;
            float edgeCost = sqrt(dx*dx + dy*dy);
            float newCost = data.h_nodes[nodeIdx].cost + edgeCost;
            
            // Update cost on host and device
            data.h_nodes[i].cost = newCost;
            CUDA_CHECK(cudaMemcpy(&data.d_nodeCost[i], &newCost, sizeof(float), cudaMemcpyHostToDevice));
            
            // Recursively update children
            updateDescendantCostsCuda(data, i);
        }
    }
}

// Function to rewire the tree using CUDA
void rewireTreeCuda(RRTStarCudaData& data, int newNodeIdx, const std::vector<int>& neighbors) {
    if (neighbors.empty()) {
        return;
    }
    
    int neighborCount = neighbors.size();
    
    // Allocate memory for device arrays
    int* d_neighbors;
    int* d_rewireFlags;
    
    CUDA_CHECK(cudaMalloc(&d_neighbors, neighborCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rewireFlags, neighborCount * sizeof(int)));
    
    // Copy neighbors to device
    CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors.data(), 
                        neighborCount * sizeof(int), cudaMemcpyHostToDevice));
    
    // Get new node data
    float newX = data.h_nodes[newNodeIdx].x;
    float newY = data.h_nodes[newNodeIdx].y;
    float newCost = data.h_nodes[newNodeIdx].cost;
    
    // Launch kernel to check which neighbors should be rewired
    int blocks = (neighborCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rewireNeighborsKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.d_nodeCost, data.d_nodeParent,
        newNodeIdx, d_neighbors, neighborCount, newX, newY, newCost, d_rewireFlags);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy rewire flags back to host
    std::vector<int> h_rewireFlags(neighborCount);
    CUDA_CHECK(cudaMemcpy(h_rewireFlags.data(), d_rewireFlags, 
                        neighborCount * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Apply rewiring and update descendants
    for (int i = 0; i < neighborCount; i++) {
        if (h_rewireFlags[i] == 1) {
            int neighborIdx = neighbors[i];
            
            // Calculate new cost
            float dx = newX - data.h_nodes[neighborIdx].x;
            float dy = newY - data.h_nodes[neighborIdx].y;
            float edgeCost = sqrt(dx*dx + dy*dy);
            float newNeighborCost = newCost + edgeCost;
            
            // Update node cost and parent on host
            data.h_nodes[neighborIdx].cost = newNeighborCost;
            data.h_nodes[neighborIdx].parent = newNodeIdx;
            
            // Update cost and parent on device
            CUDA_CHECK(cudaMemcpy(&data.d_nodeCost[neighborIdx], &newNeighborCost, 
                                sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[neighborIdx], &newNodeIdx, 
                                sizeof(int), cudaMemcpyHostToDevice));
            
            // Update costs of descendants
            updateDescendantCostsCuda(data, neighborIdx);
        }
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_rewireFlags));
}

// Function to generate random node using CUDA
void generateRandomNodeCuda(RRTStarCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax, 
                          float goalBias, float goalX, float goalY) {
    // Use pre-allocated buffers
    generateRandomNodeKernel<<<1, 1>>>(data.d_randStates, data.d_tempX, data.d_tempY, 
                                     xMin, xMax, yMin, yMax, 
                                     goalBias, goalX, goalY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(&x, data.d_tempX, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&y, data.d_tempY, sizeof(float), cudaMemcpyDeviceToHost));
}

// Function to add a new node to the tree
void addNodeCuda(RRTStarCudaData& data, float x, float y, int parent, float time, float cost) {
    if (data.h_nodeCount >= data.d_nodeCapacity) {
        std::cerr << "Error: Node capacity exceeded" << std::endl;
        return;
    }
    
    // Add to host vector for easier path extraction later
    data.h_nodes.push_back(Node(x, y, parent, time, cost));
    
    // Add to device arrays
    CUDA_CHECK(cudaMemcpy(&data.d_nodeX[data.h_nodeCount], &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeY[data.h_nodeCount], &y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[data.h_nodeCount], &parent, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeTime[data.h_nodeCount], &time, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeCost[data.h_nodeCount], &cost, sizeof(float), cudaMemcpyHostToDevice));
    
    data.h_nodeCount++;
}

// Extract path from start to goal
std::vector<Node> extractPathCuda(const RRTStarCudaData& data, int goalIndex) {
    std::vector<Node> path;
    int currentIndex = goalIndex;
    
    while (currentIndex != -1) {
        path.push_back(data.h_nodes[currentIndex]);
        currentIndex = data.h_nodes[currentIndex].parent;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

// Save tree data to file for visualization
void saveTreeToFileCuda(const RRTStarCudaData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "node_id,x,y,parent_id,time,cost" << std::endl;
    
    // Write node data
    for (int i = 0; i < data.h_nodeCount; i++) {
        file << i << ","
             << data.h_nodes[i].x << ","
             << data.h_nodes[i].y << ","
             << data.h_nodes[i].parent << ","
             << data.h_nodes[i].time << ","
             << data.h_nodes[i].cost << std::endl;
    }
    
    file.close();
    std::cout << "Tree data saved to " << filename << std::endl;
}

// Main CUDA RRT* algorithm
std::vector<Node> buildRRTStarCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
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
    int numThreads
) {
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize CUDA data
    RRTStarCudaData data;
    
    // Set world bounds
    data.h_worldXMin = xMin;
    data.h_worldXMax = xMax;
    data.h_worldYMin = yMin;
    data.h_worldYMax = yMax;
    
    // Initialize data structures
    initCudaRRTStar(data, maxIterations + 2, obstacles.size(), numThreads); // +2 for start and goal
    
    // Add start node with cost 0
    addNodeCuda(data, start.x, start.y, -1, 0.0, 0.0);
    
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
        
        // Initialize obstacle grid for faster collision checking
        initObstacleGrid(data, 64);  // 64x64 grid
    }
    
    // Track best solution
    double bestCost = std::numeric_limits<double>::infinity();
    int goalNodeIndex = -1;
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Get current time for visualization
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        float timeSeconds = static_cast<float>(elapsed.count());
        
        // Generate random node (with 5% probability, sample the goal)
        float randomX, randomY;
        generateRandomNodeCuda(data, randomX, randomY, xMin, xMax, yMin, yMax, 0.05f, goal.x, goal.y);
        
        // Skip if inside obstacle (could check using grid for speedup)
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
        
        // Find nearest node
        int nearestIndex = findNearestCuda(data, randomX, randomY);
        
        // Skip if nearest node not found (should not happen)
        if (nearestIndex < 0) {
            continue;
        }
        
        // Steer towards random node
        float newX, newY;
        float nearestX = data.h_nodes[nearestIndex].x;
        float nearestY = data.h_nodes[nearestIndex].y;
        
        float dx = randomX - nearestX;
        float dy = randomY - nearestY;
        float dist = sqrt(dx*dx + dy*dy);
        
        if (dist <= stepSize) {
            newX = randomX;
            newY = randomY;
        } else {
            float ratio = stepSize / dist;
            newX = nearestX + ratio * dx;
            newY = nearestY + ratio * dy;
        }
        
        // Check for collision
        if (checkCollisionCuda(data, nearestX, nearestY, newX, newY)) {
            continue; // Skip if collision
        }
        
        // Find nearby nodes for potential rewiring
        std::vector<int> nearNodes = findNodesInRadiusCuda(data, newX, newY, rewireRadius);
        
        // Choose best parent
        int bestParentIndex = chooseBestParentCuda(data, newX, newY, nearNodes);
        
        if (bestParentIndex < 0) {
            continue; // Skip if no valid parent found
        }
        
        // Calculate cost from start to this node
        float parentX = data.h_nodes[bestParentIndex].x;
        float parentY = data.h_nodes[bestParentIndex].y;
        float parentCost = data.h_nodes[bestParentIndex].cost;
        float edgeCost = sqrt((newX - parentX)*(newX - parentX) + (newY - parentY)*(newY - parentY));
        float nodeCost = parentCost + edgeCost;
        
        // Add new node
        addNodeCuda(data, newX, newY, bestParentIndex, timeSeconds, nodeCost);
        int newNodeIndex = data.h_nodeCount - 1;
        
        // Rewire the tree to optimize paths
        rewireTreeCuda(data, newNodeIndex, nearNodes);
        
        // Check if goal reached
        dx = newX - goal.x;
        dy = newY - goal.y;
        float distToGoal = sqrt(dx*dx + dy*dy);
        
        if (distToGoal <= goalThreshold) {
            // Check if path to goal is collision-free
            if (!checkCollisionCuda(data, newX, newY, goal.x, goal.y)) {
                // Calculate total cost to goal
                float edgeCostToGoal = sqrt(dx*dx + dy*dy);
                float totalCost = nodeCost + edgeCostToGoal;
                
                // Check if this is a better path
                if (totalCost < bestCost) {
                    bestCost = totalCost;
                    
                    // Add or update goal node with improved cost
                    if (goalNodeIndex >= 0) {
                        // Update existing goal node
                        data.h_nodes[goalNodeIndex].parent = newNodeIndex;
                        data.h_nodes[goalNodeIndex].cost = totalCost;
                        data.h_nodes[goalNodeIndex].time = timeSeconds;
                        
                        // Update on device
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[goalNodeIndex], &newNodeIndex, sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeCost[goalNodeIndex], &totalCost, sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeTime[goalNodeIndex], &timeSeconds, sizeof(float), cudaMemcpyHostToDevice));
                    } else {
                        // Add new goal node
                        addNodeCuda(data, goal.x, goal.y, newNodeIndex, timeSeconds, totalCost);
                        goalNodeIndex = data.h_nodeCount - 1;
                    }
                    
                    std::cout << "Improved solution found with cost: " << bestCost << std::endl;
                }
            }
        }
        
        // Periodic visualization
        if (enableVisualization && i % 100 == 0) {
            saveTreeToFileCuda(data, treeFilename);
        }
    }
    
    // Save final tree data if visualization is enabled
    if (enableVisualization) {
        saveTreeToFileCuda(data, treeFilename);
    }
    
    // Extract path if goal was reached
    std::vector<Node> path;
    if (goalNodeIndex >= 0) {
        path = extractPathCuda(data, goalNodeIndex);
        
        // Print final solution information
        std::cout << "Final solution cost: " << bestCost << std::endl;
        std::cout << "Path length: " << path.size() << " nodes" << std::endl;
    } else {
        std::cout << "Goal not reached within max iterations." << std::endl;
    }
    
    // Clean up CUDA resources
    cleanupCudaRRTStar(data);
    
    return path;
}