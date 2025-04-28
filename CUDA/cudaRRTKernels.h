#ifndef CUDA_RRT_KERNELS_H
#define CUDA_RRT_KERNELS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>
#include "cudaRRTUtils.h"

// CUDA kernel to initialize random states
__global__ static void initRandStatesKernel(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel to find the nearest node to a query point
__global__ static void findNearestKernel(float* nodeX, float* nodeY, int nodeCount, 
                                float queryX, float queryY, 
                                float* distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        float dx = nodeX[idx] - queryX;
        float dy = nodeY[idx] - queryY;
        distances[idx] = dx*dx + dy*dy; // Squared distance (faster than sqrt)
    }
}

// CUDA kernel to find nearest node in the specified tree
__global__ static void findNearestInTreeKernel(float* nodeX, float* nodeY, int* nodeTree, int nodeCount,
                                      float queryX, float queryY, int treeIdx,
                                      float* distances, int* validNode) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < nodeCount) {
        // Mark if this node belongs to the specified tree
        validNode[idx] = (nodeTree[idx] == treeIdx) ? 1 : 0;
        
        if (validNode[idx]) {
            // Calculate squared distance
            float dx = nodeX[idx] - queryX;
            float dy = nodeY[idx] - queryY;
            distances[idx] = dx*dx + dy*dy;
        } else {
            // Set a large distance for nodes not in the specified tree
            distances[idx] = FLT_MAX;
        }
    }
}

// CUDA kernel to find closest pairs between trees
__global__ static void findClosestPairsKernel(float* nodeX, float* nodeY, int* nodeTree, int nodeCount,
                                     float* distanceMatrix, int startTreeSize, int goalTreeSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalPairs = startTreeSize * goalTreeSize;
    
    if (idx < totalPairs) {
        // Convert linear index to start and goal tree indices
        int startMatrixIdx = idx / goalTreeSize;
        int goalMatrixIdx = idx % goalTreeSize;
        
        // Find actual node indices
        int startNodeIdx = -1;
        int goalNodeIdx = -1;
        int startCount = 0;
        int goalCount = 0;
        
        for (int i = 0; i < nodeCount && (startNodeIdx == -1 || goalNodeIdx == -1); i++) {
            if (nodeTree[i] == 0) { // Start tree
                if (startCount == startMatrixIdx) {
                    startNodeIdx = i;
                }
                startCount++;
            } else if (nodeTree[i] == 1) { // Goal tree
                if (goalCount == goalMatrixIdx) {
                    goalNodeIdx = i;
                }
                goalCount++;
            }
        }
        
        // Calculate distance between nodes
        if (startNodeIdx != -1 && goalNodeIdx != -1) {
            float dx = nodeX[startNodeIdx] - nodeX[goalNodeIdx];
            float dy = nodeY[startNodeIdx] - nodeY[goalNodeIdx];
            distanceMatrix[idx] = dx*dx + dy*dy;
        } else {
            distanceMatrix[idx] = FLT_MAX;
        }
    }
}

// CUDA kernel to check collision with obstacles
__global__ static void checkCollisionKernel(float x1, float y1, float x2, float y2,
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
            atomicExch((int*)&sharedCollision, 1);
        }
    }
    __syncthreads();
    
    // First thread writes the result
    if (threadIdx.x == 0) {
        *collisionResult = sharedCollision;
    }
}

// CUDA kernel to find nodes within a radius
__global__ static void findNodesInRadiusKernel(float* nodeX, float* nodeY, int nodeCount,
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
__global__ static void findBestParentKernel(float* nodeX, float* nodeY, float* nodeCost,
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
        
        // Use atomicMinFloat to update best parent
        atomicMinFloat(minCost, totalCost, bestParent, nodeIdx);
    }
}

// CUDA kernel for tree rewiring
__global__ static void rewireNeighborsKernel(float* nodeX, float* nodeY, float* nodeCost,
                                    int* nodeParent, int nodeCount,
                                    int* neighbors, int neighborCount,
                                    float newX, float newY, float newCost,
                                    int* updated) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < neighborCount) {
        int nodeIdx = neighbors[idx];
        
        // Skip the new node's parent
        if (nodeIdx == nodeParent[nodeCount-1]) return;
        
        float dx = nodeX[nodeIdx] - newX;
        float dy = nodeY[nodeIdx] - newY;
        float edgeCost = sqrtf(dx*dx + dy*dy);
        float potentialCost = newCost + edgeCost;
        
        // If rewiring would improve cost, update
        if (potentialCost < nodeCost[nodeIdx]) {
            atomicExch(&nodeParent[nodeIdx], nodeCount-1);
            atomicExch(&nodeCost[nodeIdx], potentialCost);
            atomicExch(&updated[idx], 1);
        }
    }
}

// CUDA kernel to generate random nodes (for standard RRT)
__global__ static void generateRandomNodeKernel(curandState* randStates, 
                                      float* x, float* y,
                                      float xMin, float xMax, 
                                      float yMin, float yMax,
                                      float goalBias=0.0, float goalX=0.0, float goalY=0.0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Generate a random number to determine if we should bias toward the goal
    float r = curand_uniform(&randStates[idx]);
    
    if (r < goalBias) {
        // Use goal position
        x[idx] = goalX;
        y[idx] = goalY;
    } else {
        // Generate random position in bounds
        x[idx] = xMin + curand_uniform(&randStates[idx]) * (xMax - xMin);
        y[idx] = yMin + curand_uniform(&randStates[idx]) * (yMax - yMin);
    }
}

// CUDA helper function for steering (shared across implementations)
__host__ __device__ static void steerCuda(float x1, float y1, float x2, float y2, float stepSize,
                                 float& newX, float& newY) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx*dx + dy*dy);
    
    if (distance <= stepSize) {
        // If destination is closer than step size, go directly there
        newX = x2;
        newY = y2;
    } else {
        // Otherwise, move in the direction of the destination by stepSize
        float ratio = stepSize / distance;
        newX = x1 + dx * ratio;
        newY = y1 + dy * ratio;
    }
}

// CUDA kernel to sample from the informed subset for RRT* Informed
__global__ static void sampleInformedSubsetKernel(curandState* randStates,
                                         float startX, float startY, float goalX, float goalY,
                                         float currentBestCost, float goalBias,
                                         float xMin, float xMax, float yMin, float yMax,
                                         float* x, float* y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate ellipse parameters
    float cMax = currentBestCost; // Current best path length
    float dist = sqrtf((goalX - startX) * (goalX - startX) + (goalY - startY) * (goalY - startY));
    float a = cMax / 2.0f; // Semi-major axis
    float c = dist / 2.0f; // Half the distance between foci
    float b = sqrtf(a*a - c*c); // Semi-minor axis
    
    // Calculate center of the ellipse
    float centerX = (startX + goalX) / 2.0f;
    float centerY = (startY + goalY) / 2.0f;
    
    // Calculate rotation angle of the ellipse
    float angle = atan2f(goalY - startY, goalX - startX);
    
    // Apply goal bias
    float r = curand_uniform(&randStates[idx]);
    if (r < goalBias) {
        // Use goal position
        *x = goalX;
        *y = goalY;
        return;
    }
    
    // Sample from the ellipse
    float t = 2.0f * M_PI * curand_uniform(&randStates[idx]); // Random angle [0, 2π]
    float u = curand_uniform(&randStates[idx]) + curand_uniform(&randStates[idx]); // Random radius factor
    float r2 = (u > 1.0f) ? 2.0f - u : u; // Transform to get uniform distribution in the ellipse
    
    // Calculate point in the ellipse local coordinate system
    float localX = a * r2 * cosf(t);
    float localY = b * r2 * sinf(t);
    
    // Rotate and translate to world coordinates
    *x = centerX + localX * cosf(angle) - localY * sinf(angle);
    *y = centerY + localX * sinf(angle) + localY * cosf(angle);
    
    // Clamp to world bounds
    *x = fmaxf(xMin, fminf(xMax, *x));
    *y = fmaxf(yMin, fminf(yMax, *y));
}

#endif // CUDA_RRT_KERNELS_H