// rrtCuda.cu
#include "rrtCuda.h"
#include "cudaRRTKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

// Destructor implementation
RRTCudaData::~RRTCudaData() {
    // Free device memory
    if (d_nodeX) cudaFree(d_nodeX);
    if (d_nodeY) cudaFree(d_nodeY);
    if (d_nodeParent) cudaFree(d_nodeParent);
    if (d_nodeTime) cudaFree(d_nodeTime);
    
    if (d_obstacleX) cudaFree(d_obstacleX);
    if (d_obstacleY) cudaFree(d_obstacleY);
    if (d_obstacleWidth) cudaFree(d_obstacleWidth);
    if (d_obstacleHeight) cudaFree(d_obstacleHeight);
    
    if (d_randStates) cudaFree(d_randStates);
}

// Kernel implementations are moved to cudaRRTKernels.h

// Function to initialize CUDA resources
void initCudaRRT(RRTCudaData& data, int maxNodes, int numObstacles, int numThreads) {
    // Allocate memory for nodes
    data.d_nodeCapacity = maxNodes;
    CUDA_CHECK(cudaMalloc(&data.d_nodeX, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeY, maxNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeParent, maxNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_nodeTime, maxNodes * sizeof(float)));
    
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
}

// Function to clean up CUDA resources
void cleanupCudaRRT(RRTCudaData& data) {
    // Most cleanup is handled by the destructor
    // This function is provided for explicit cleanup if needed
    cudaDeviceSynchronize();
}

// Function to find nearest node using CUDA
int findNearestCuda(RRTCudaData& data, float x, float y) {
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

// Function to check collision using CUDA
bool checkCollisionCuda(RRTCudaData& data, float x1, float y1, float x2, float y2) {
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

// Function to generate random node using CUDA
void generateRandomNodeCuda(RRTCudaData& data, float& x, float& y, 
                          float xMin, float xMax, float yMin, float yMax, 
                          float goalBias, float goalX, float goalY) {
    // Allocate memory for result
    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
    
    // Launch kernel to generate random node
    // Using just one thread since we need only one random node
    generateRandomNodeKernel<<<1, 1>>>(data.d_randStates, d_x, d_y, 
                                     xMin, xMax, yMin, yMax, 
                                     goalBias, goalX, goalY);
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
void addNodeCuda(RRTCudaData& data, float x, float y, int parent, float time) {
    if (data.h_nodeCount >= data.d_nodeCapacity) {
        std::cerr << "Error: Node capacity exceeded" << std::endl;
        return;
    }
    
    // Add to host vectors for easier path extraction later
    data.h_nodes.push_back(Node(x, y, parent, time));
    
    // Add to device arrays
    CUDA_CHECK(cudaMemcpy(&data.d_nodeX[data.h_nodeCount], &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeY[data.h_nodeCount], &y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeParent[data.h_nodeCount], &parent, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data.d_nodeTime[data.h_nodeCount], &time, sizeof(float), cudaMemcpyHostToDevice));
    
    data.h_nodeCount++;
}

// Steer function is defined in cudaRRTKernels.h

// Extract path from start to goal
std::vector<Node> extractPathCuda(const RRTCudaData& data, int goalIndex) {
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
void saveTreeToFileCuda(const RRTCudaData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "node_id,x,y,parent_id,time" << std::endl;
    
    // Write node data
    for (int i = 0; i < data.h_nodeCount; i++) {
        file << i << ","
             << data.h_nodes[i].x << ","
             << data.h_nodes[i].y << ","
             << data.h_nodes[i].parent << ","
             << data.h_nodes[i].time << std::endl;
    }
    
    file.close();
    std::cout << "Tree data saved to " << filename << std::endl;
}

// Main CUDA RRT algorithm
std::vector<Node> buildRRTCuda(
    const Node& start,
    const Node& goal,
    const std::vector<Obstacle>& obstacles,
    double stepSize,
    double goalThreshold,
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
    RRTCudaData data;
    initCudaRRT(data, maxIterations + 2, obstacles.size(), numThreads); // +2 for start and goal
    
    // Add start node
    addNodeCuda(data, start.x, start.y, -1, 0.0);
    
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
        
        // Generate random node (with small probability, sample the goal)
        float randomX, randomY;
        generateRandomNodeCuda(data, randomX, randomY, xMin, xMax, yMin, yMax, 0.05f, goal.x, goal.y);
        
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
        
        // Add new node
        addNodeCuda(data, newX, newY, nearestIndex, timeSeconds);
        int newNodeIndex = data.h_nodeCount - 1;
        
        // Check if goal reached
        float dx = newX - goal.x;
        float dy = newY - goal.y;
        float distToGoal = sqrtf(dx*dx + dy*dy);
        
        if (distToGoal <= goalThreshold) {
            // Check if path to goal is collision-free
            if (!checkCollisionCuda(data, newX, newY, goal.x, goal.y)) {
                // Add goal node
                addNodeCuda(data, goal.x, goal.y, newNodeIndex, timeSeconds);
                
                // Save tree data if visualization is enabled
                if (enableVisualization) {
                    saveTreeToFileCuda(data, treeFilename);
                }
                
                // Extract path
                std::vector<Node> path = extractPathCuda(data, data.h_nodeCount - 1);
                
                // Clean up CUDA resources
                cleanupCudaRRT(data);
                
                return path;
            }
        }
    }
    
    // If goal not reached, save tree anyway if visualization is enabled
    if (enableVisualization) {
        saveTreeToFileCuda(data, treeFilename);
    }
    
    // Clean up CUDA resources
    cleanupCudaRRT(data);
    
    // If goal not reached, return empty path
    std::cout << "Goal not reached within max iterations." << std::endl;
    return std::vector<Node>();
}