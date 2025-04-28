// cudaRRTUtils.h
// Shared utility functions for CUDA-accelerated RRT implementations

#ifndef CUDA_RRT_UTILS_H
#define CUDA_RRT_UTILS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>

// CUDA block size definition
#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Atomic minimum update with float key and tracking of best index
__device__ inline void atomicMinFloat(float* minCost, float cost, int* bestIndex, int index) {
    // Only attempt to update if this cost is better than current minimum
    if (cost < *minCost) {
        unsigned int* address_as_uint = (unsigned int*)minCost;
        unsigned int old = __float_as_uint(*minCost);
        unsigned int assumed;
        
        do {
            assumed = old;
            // Convert float bit pattern back to float for comparison
            if (__uint_as_float(assumed) <= cost) break;
            old = atomicCAS(address_as_uint, assumed, __float_as_uint(cost));
        } while (assumed != old);
        
        // If we successfully updated the cost, also update the index
        if (__uint_as_float(old) > cost) {
            atomicExch(bestIndex, index);
        }
    }
}

// Euclidean distance calculation (device function)
__device__ inline float distanceCuda(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

// Euclidean distance squared (faster when only comparing distances)
__device__ inline float distanceSquaredCuda(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx*dx + dy*dy;
}

// Check if a point is inside an obstacle (device function)
__device__ inline bool isPointInObstacle(float x, float y, 
                                float obsX, float obsY, 
                                float obsWidth, float obsHeight) {
    return (x >= obsX && x <= obsX + obsWidth && 
            y >= obsY && y <= obsY + obsHeight);
}

#endif // CUDA_RRT_UTILS_H