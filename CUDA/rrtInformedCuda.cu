// rrtInformedCuda.cu
#include "rrtInformedCuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
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
RRTInformedCudaData::~RRTInformedCudaData() {
    // Free device memory
    if (d_nodeX) cudaFree(d_nodeX);
    if (d_nodeY) cudaFree(d_nodeY);
    if (d_nodeParent) cudaFree(d_nodeParent);
    if (d_nodeTime) cudaFree(d_nodeTime);
    if (d_nodeCost) cudaFree(d_nodeCost);
    
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

// CUDA kernel to find the nearest node to a query point
__global__ void findNearestKernel(float* nodeX, float* nodeY, int nodeCount, 
                                float queryX, float queryY, 
                                float* distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        float dx = nodeX[idx] - queryX;
        float dy = nodeY[idx] - queryY;
        distances[idx] = dx*dx + dy*dy; // Squared distance (faster than sqrt)
    }
}

// CUDA kernel to find nodes within a radius
__global__ void findNodesInRadiusKernel(float* nodeX, float* nodeY, int nodeCount,
                                      float queryX, float queryY, float radiusSq,
                                      int* inRadius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        float dx = nodeX[idx] - queryX;
        float dy = nodeY[idx] - queryY;
        float distSq = dx*dx + dy*dy;
        
        // Mark as 1 if within radius, 0 otherwise
        inRadius[idx] = (distSq <= radiusSq) ? 1 : 0;
    }
}

// CUDA kernel for best parent selection
__global__ void findBestParentKernel(float* nodeX, float* nodeY, float* nodeCost,
                                   int* neighbors, int neighborCount,
                                   float newX, float newY,
                                   float* costToNew, float* minCost, int* bestParent) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < neighborCount) {
        int nodeIdx = neighbors[idx];
        float dx = nodeX[nodeIdx] - newX;
        float dy = nodeY[nodeIdx] - newY;
        float edgeCost = sqrtf(dx*dx + dy*dy);
        float totalCost = nodeCost[nodeIdx] + edgeCost;
        
        costToNew[idx] = totalCost;
        
        // Atomic minimum update for best parent
        atomicMinFloat(minCost, totalCost, bestParent, nodeIdx);
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

// CUDA kernel for rewiring neighbors
__global__ void rewireNeighborsKernel(float* nodeX, float* nodeY, float* nodeCost,
                                    int* nodeParent, int newNodeIdx,
                                    int* neighbors, int neighborCount,
                                    float newX, float newY, float newCost,
                                    int* rewireFlags) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < neighborCount) {
        int neighborIdx = neighbors[idx];
        
        // Skip if this is the parent of the new node
        if (neighborIdx == nodeParent[newNodeIdx]) {
            rewireFlags[idx] = 0;
            return;
        }
        
        // Calculate cost through new node
        float dx = nodeX[neighborIdx] - newX;
        float dy = nodeY[neighborIdx] - newY;
        float edgeCost = sqrtf(dx*dx + dy*dy);
        float costThroughNew = newCost + edgeCost;
        
        // Check if cheaper path found
        if (costThroughNew < nodeCost[neighborIdx]) {
            rewireFlags[idx] = 1;
        } else {
            rewireFlags[idx] = 0;
        }
    }
}

// CUDA kernel for sampling unit ball
__global__ void sampleUnitBallKernel(curandState* randStates, float* x, float* y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    float rx, ry, magSq;
    do {
        rx = 2.0f * curand_uniform(&randStates[idx]) - 1.0f;
        ry = 2.0f * curand_uniform(&randStates[idx]) - 1.0f;
        magSq = rx*rx + ry*ry;
    } while (magSq > 1.0f || magSq < 0.000001f);
    
    *x = rx;
    *y = ry;
}

// CUDA kernel to transform unit ball sample to informed ellipsoid
__global__ void transformToEllipsoidKernel(float* unitX, float* unitY,
                                        float startX, float startY, float goalX, float goalY,
                                        float bestCost, float* transformedX, float* transformedY,
                                        float xMin, float xMax, float yMin, float yMax) {
    // Calculate ellipsoid parameters
    float centerX = (startX + goalX) / 2.0f;
    float centerY = (startY + goalY) / 2.0f;
    
    // Calculate minimum cost (Euclidean distance from start to goal)
    float dx = goalX - startX;
    float dy = goalY - startY;
    float cmin = sqrtf(dx*dx + dy*dy);
    
    // Transverse axis is the best cost
    float transverseAxis = bestCost;
    
    // Calculate conjugate axis
    float conjugateAxis = 0.0f;
    if (bestCost > cmin) {
        conjugateAxis = sqrtf(bestCost * bestCost - cmin * cmin);
    } else {
        // If not valid (best cost too small), just return a point in the original space
        *transformedX = xMin + (*unitX + 1.0f) * 0.5f * (xMax - xMin);
        *transformedY = yMin + (*unitY + 1.0f) * 0.5f * (yMax - yMin);
        return;
    }
    
    // If conjugate axis is too small, sample from original space
    if (conjugateAxis < 0.001f) {
        *transformedX = xMin + (*unitX + 1.0f) * 0.5f * (xMax - xMin);
        *transformedY = yMin + (*unitY + 1.0f) * 0.5f * (yMax - yMin);
        return;
    }
    
    // Calculate rotation matrix (from ellipsoid to world frame)
    float a1_x = (goalX - startX) / cmin;  // Normalized x direction
    float a1_y = (goalY - startY) / cmin;  // Normalized y direction
    
    // Perpendicular direction for second axis
    float a2_x = -a1_y;
    float a2_y = a1_x;
    
    // Scaling factors
    float r1 = transverseAxis / 2.0f;  // Transverse radius
    float r2 = conjugateAxis / 2.0f;   // Conjugate radius
    
    // Transform: x_ellipse = C * L * x_ball + center
    // Scale input point
    float scaledX = r1 * (*unitX);
    float scaledY = r2 * (*unitY);
    
    // Rotate and translate
    *transformedX = a1_x * scaledX + a2_x * scaledY + centerX;
    *transformedY = a1_y * scaledX + a2_y * scaledY + centerY;
    
    // Clamp to bounds
    *transformedX = fmaxf(xMin, fminf(xMax, *transformedX));
    *transformedY = fmaxf(yMin, fminf(yMax, *transformedY));
}

// CUDA kernel for sampling in the informed subset
__global__ void sampleInformedSubsetKernel(curandState* randStates, 
                                        float startX, float startY, float goalX, float goalY,
                                        float bestCost, float goalBias,
                                        float xMin, float xMax, float yMin, float yMax,
                                        float* sampledX, float* sampledY) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // With probability goalBias, return the goal
    float r = curand_uniform(&randStates[idx]);
    if (r < goalBias) {
        *sampledX = goalX;
        *sampledY = goalY;
        return;
    }
    
    // Calculate if we have a valid best cost to use for informed sampling
    float dx = goalX - startX;
    float dy = goalY - startY;
    float cmin = sqrtf(dx*dx + dy*dy);
    
    // If we don't have a valid solution yet or the best cost is close to optimal, 
    // sample from the original space
    if (bestCost >= FLT_MAX || bestCost <= cmin * 1.01f) {
        *sampledX = xMin + curand_uniform(&randStates[idx]) * (xMax - xMin);
        *sampledY = yMin + curand_uniform(&randStates[idx]) * (yMax - yMin);
        return;
    }
    
    // Sample from unit ball
    float unitX, unitY, magSq;
    do {
        unitX = 2.0f * curand_uniform(&randStates[idx]) - 1.0f;
        unitY = 2.0f * curand_uniform(&randStates[idx]) - 1.0f;
        magSq = unitX*unitX + unitY*unitY;
    } while (magSq > 1.0f || magSq < 0.000001f);
    
    // Transform to ellipsoid
    float centerX = (startX + goalX) / 2.0f;
    float centerY = (startY + goalY) / 2.0f;
    
    // Calculate rotation matrix
    float a1_x = (goalX - startX) / cmin;  // Normalized x direction
    float a1_y = (goalY - startY) / cmin;  // Normalized y direction
    
    // Perpendicular direction for second axis
    float a2_x = -a1_y;
    float a2_y = a1_x;
    
    // Calculate conjugate axis length
    float conjugateAxis = sqrtf(bestCost * bestCost - cmin * cmin);
    
    // Scaling factors
    float r1 = bestCost / 2.0f;        // Transverse radius
    float r2 = conjugateAxis / 2.0f;   // Conjugate radius
    
    // Scale input point
    float scaledX = r1 * unitX;
    float scaledY = r2 * unitY;
    
    // Rotate and translate
    *sampledX = a1_x * scaledX + a2_x * scaledY + centerX;
    *sampledY = a1_y * scaledX + a2_y * scaledY + centerY;
    
    // Clamp to bounds
    *sampledX = fmaxf(xMin, fminf(xMax, *sampledX));
    *sampledY = fmaxf(yMin, fminf(yMax, *sampledY));
}


// Function to initialize CUDA resources
void initCudaRRTInformed(RRTInformedCudaData& data, int maxNodes, int numObstacles, int numThreads) {
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
    
    // Initialize random states
    CUDA_CHECK(cudaMalloc(&data.d_randStates, numThreads * sizeof(curandState)));
    
    int blocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandStatesKernel<<<blocks, BLOCK_SIZE>>>(data.d_randStates, 
                                               static_cast<unsigned long>(time(nullptr)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize best cost to infinity
    data.h_bestCost = FLT_MAX;
    data.h_goalNodeIndex = -1;
}

// Function to clean up CUDA resources
void cleanupCudaRRTInformed(RRTInformedCudaData& data) {
    // Most cleanup is handled by the destructor
    // This function is provided for explicit cleanup if needed
    cudaDeviceSynchronize();
}

// Function to find nearest node using CUDA
int findNearestCuda(RRTInformedCudaData& data, float x, float y) {
    if (data.h_nodeCount == 0) {
        return -1;
    }
    
    // Allocate memory for distances
    float* d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, data.h_nodeCount * sizeof(float)));
    
    // Calculate number of blocks needed
    int blocks = (data.h_nodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to compute distances
    findNearestKernel<<<blocks, BLOCK_SIZE>>>(data.d_nodeX, data.d_nodeY, data.h_nodeCount, 
                                          x, y, d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Find minimum distance index using Thrust
    thrust::device_ptr<float> thrust_distances(d_distances);
    thrust::device_ptr<float> min_distance_ptr = thrust::min_element(
        thrust_distances, thrust_distances + data.h_nodeCount);
    
    // Get the index of the minimum element
    int minIndex = static_cast<int>(min_distance_ptr - thrust_distances);
    
    // Free memory
    CUDA_CHECK(cudaFree(d_distances));
    
    return minIndex;
}

// Function to find nodes within a radius using CUDA
std::vector<int> findNodesInRadiusCuda(RRTInformedCudaData& data, float x, float y, float radius) {
    std::vector<int> result;
    
    if (data.h_nodeCount == 0) {
        return result;
    }
    
    // Allocate memory for results
    int* d_inRadius;
    CUDA_CHECK(cudaMalloc(&d_inRadius, data.h_nodeCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_inRadius, 0, data.h_nodeCount * sizeof(int)));
    
    // Calculate number of blocks needed
    int blocks = (data.h_nodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel to find nodes in radius
    float radiusSq = radius * radius;
    findNodesInRadiusKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.h_nodeCount,
        x, y, radiusSq, d_inRadius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<int> h_inRadius(data.h_nodeCount);
    CUDA_CHECK(cudaMemcpy(h_inRadius.data(), d_inRadius, 
                        data.h_nodeCount * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Build result vector
    for (int i = 0; i < data.h_nodeCount; i++) {
        if (h_inRadius[i] == 1) {
            result.push_back(i);
        }
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_inRadius));
    
    return result;
}

// Function to choose best parent using CUDA
int chooseBestParentCuda(RRTInformedCudaData& data, float x, float y, 
                       const std::vector<int>& neighbors) {
    if (neighbors.empty()) {
        return -1;
    }
    
    int neighborCount = neighbors.size();
    
    // Allocate memory for device arrays
    int* d_neighbors;
    float* d_costToNew;
    float* d_minCost;
    int* d_bestParent;
    
    CUDA_CHECK(cudaMalloc(&d_neighbors, neighborCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_costToNew, neighborCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_minCost, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bestParent, sizeof(int)));
    
    // Initialize best cost to a large value and best parent to -1
    float initialCost = FLT_MAX;
    int initialParent = -1;
    CUDA_CHECK(cudaMemcpy(d_minCost, &initialCost, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bestParent, &initialParent, sizeof(int), cudaMemcpyHostToDevice));
    
    // Copy neighbors to device
    CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors.data(), 
                        neighborCount * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel to find best parent
    int blocks = (neighborCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findBestParentKernel<<<blocks, BLOCK_SIZE>>>(
        data.d_nodeX, data.d_nodeY, data.d_nodeCost,
        d_neighbors, neighborCount, x, y, d_costToNew, d_minCost, d_bestParent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get result
    int bestParent;
    CUDA_CHECK(cudaMemcpy(&bestParent, d_bestParent, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free memory
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_costToNew));
    CUDA_CHECK(cudaFree(d_minCost));
    CUDA_CHECK(cudaFree(d_bestParent));
    
    return bestParent;
}

// Function to check collision using CUDA
bool checkCollisionCuda(RRTInformedCudaData& data, float x1, float y1, float x2, float y2) {
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

// Function to rewire the tree using CUDA
void rewireTreeCuda(RRTInformedCudaData& data, int newNodeIdx, const std::vector<int>& neighbors) {
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
    float newX, newY, newCost;
    CUDA_CHECK(cudaMemcpy(&newX, &data.d_nodeX[newNodeIdx], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&newY, &data.d_nodeY[newNodeIdx], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&newCost, &data.d_nodeCost[newNodeIdx], sizeof(float), cudaMemcpyDeviceToHost));
    
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
    
    // Apply rewiring on the host (for simplicity)
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
            
            // TODO: Recursive rewiring of descendants (if needed)
            // This would require a separate function to propagate cost changes
        }
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_rewireFlags));
}

// Function to sample from the informed subset
void sampleInformedSubsetCuda(RRTInformedCudaData& data, float& x, float& y, 
                             float xMin, float xMax, float yMin, float yMax, float goalBias) {
    // Allocate memory for result
    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
    
    // Launch kernel to generate random node in the informed subset
    sampleInformedSubsetKernel<<<1, 1>>>(
        data.d_randStates,
        data.h_startX, data.h_startY, data.h_goalX, data.h_goalY,
        data.h_bestCost, goalBias,
        xMin, xMax, yMin, yMax,
        d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(&x, d_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free memory
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// Function to add a new node to the tree
void addNodeCuda(RRTInformedCudaData& data, float x, float y, int parent, float time, float cost) {
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

// Update best cost when a better solution is found
void updateBestCostCuda(RRTInformedCudaData& data, float newCost, int goalNodeIndex) {
    if (newCost < data.h_bestCost) {
        data.h_bestCost = newCost;
        data.h_goalNodeIndex = goalNodeIndex;
    }
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

// Extract path from start to goal
std::vector<Node> extractPathCuda(const RRTInformedCudaData& data, int goalIndex) {
    std::vector<Node> path;
    
    if (goalIndex < 0 || goalIndex >= data.h_nodeCount) {
        return path; // Empty path if goal index is invalid
    }
    
    int currentIndex = goalIndex;
    
    while (currentIndex != -1) {
        path.push_back(data.h_nodes[currentIndex]);
        currentIndex = data.h_nodes[currentIndex].parent;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

// Save tree data to file for visualization
void saveTreeToFileCuda(const RRTInformedCudaData& data, const std::string& filename) {
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

// Main CUDA Informed RRT* algorithm
std::vector<Node> buildRRTInformedCuda(
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
    int numThreads,
    bool stopAtFirstSolution
) {
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize CUDA data
    RRTInformedCudaData data;
    initCudaRRTInformed(data, maxIterations + 2, obstacles.size(), numThreads); // +2 for start and goal
    
    // Store start and goal coordinates
    data.h_startX = start.x;
    data.h_startY = start.y;
    data.h_goalX = goal.x;
    data.h_goalY = goal.y;
    
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
    }
    
    // Main loop
    for (int i = 0; i < maxIterations; i++) {
        // Get current time for visualization
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        float timeSeconds = static_cast<float>(elapsed.count());
        
        // Sample from the informed subset (5% goal bias)
        float randomX, randomY;
        sampleInformedSubsetCuda(data, randomX, randomY, xMin, xMax, yMin, yMax, 0.05f);
        
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
        
        // Find nearest node
        int nearestIndex = findNearestCuda(data, randomX, randomY);
        
        // Skip if nearest node not found (should not happen)
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
        
        // Find nearby nodes for potential rewiring
        std::vector<int> nearNodes = findNodesInRadiusCuda(data, newX, newY, rewireRadius);
        
        // Choose best parent
        int bestParentIndex = chooseBestParentCuda(data, newX, newY, nearNodes);
        
        if (bestParentIndex < 0) {
            continue; // Skip if no valid parent found
        }
        
        // Calculate cost from start to this node
        float edgeCost = sqrtf(
            (newX - data.h_nodes[bestParentIndex].x) * (newX - data.h_nodes[bestParentIndex].x) +
            (newY - data.h_nodes[bestParentIndex].y) * (newY - data.h_nodes[bestParentIndex].y)
        );
        float nodeCost = data.h_nodes[bestParentIndex].cost + edgeCost;
        
        // Add new node
        addNodeCuda(data, newX, newY, bestParentIndex, timeSeconds, nodeCost);
        int newNodeIndex = data.h_nodeCount - 1;
        
        // Rewire the tree to optimize paths
        rewireTreeCuda(data, newNodeIndex, nearNodes);
        
        // Check if goal reached
        float dx = newX - goal.x;
        float dy = newY - goal.y;
        float distToGoal = sqrtf(dx*dx + dy*dy);
        
        if (distToGoal <= goalThreshold) {
            // Check if path to goal is collision-free
            if (!checkCollisionCuda(data, newX, newY, goal.x, goal.y)) {
                // Calculate total cost to goal
                float edgeCostToGoal = sqrtf(dx*dx + dy*dy);
                float totalCost = nodeCost + edgeCostToGoal;
                
                // Check if this is a better path
                if (totalCost < data.h_bestCost) {
                    // Add or update goal node with improved cost
                    if (data.h_goalNodeIndex >= 0) {
                        // Update existing goal node
                        data.h_nodes[data.h_goalNodeIndex].parent = newNodeIndex;
                        data.h_nodes[data.h_goalNodeIndex].cost = totalCost;
                        data.h_nodes[data.h_goalNodeIndex].time = timeSeconds;
                        
                        // Update on device
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[data.h_goalNodeIndex], &newNodeIndex, sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeCost[data.h_goalNodeIndex], &totalCost, sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(&data.d_nodeTime[data.h_goalNodeIndex], &timeSeconds, sizeof(float), cudaMemcpyHostToDevice));
                    } else {
                        // Add new goal node
                        addNodeCuda(data, goal.x, goal.y, newNodeIndex, timeSeconds, totalCost);
                        data.h_goalNodeIndex = data.h_nodeCount - 1;
                    }
                    
                    // Update best cost
                    updateBestCostCuda(data, totalCost, data.h_goalNodeIndex);
                    
                    std::cout << "Improved solution found with cost: " << totalCost << std::endl;
                    
                    // Stop at first solution if requested
                    if (stopAtFirstSolution) {
                        // Save the tree data if visualization is enabled
                        if (enableVisualization) {
                            saveTreeToFileCuda(data, treeFilename);
                        }
                        
                        std::cout << "Stopping at first solution as requested." << std::endl;
                        
                        // Extract and return path
                        std::vector<Node> path = extractPathCuda(data, data.h_goalNodeIndex);
                        
                        // Clean up CUDA resources
                        cleanupCudaRRTInformed(data);
                        
                        return path;
                    }
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
    if (data.h_goalNodeIndex >= 0) {
        path = extractPathCuda(data, data.h_goalNodeIndex);
        
        // Print final solution information
        std::cout << "Final solution cost: " << data.h_bestCost << std::endl;
        std::cout << "Path length: " << path.size() << " nodes" << std::endl;
    } else {
        std::cout << "Goal not reached within max iterations." << std::endl;
    }
    
    // Clean up CUDA resources
    cleanupCudaRRTInformed(data);
    
    return path;
}