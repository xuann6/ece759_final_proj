#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>      // For sqrtf, floorf
#include <limits>     // For HUGE_VALF
#include <chrono>
#include <algorithm>  // For std::max, std::min, std::reverse
#include <curand_kernel.h> // Include cuRAND for GPU random numbers
#include <math.h>     // For ceilf

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Parameters ---
// World parameters
#define WORLD_WIDTH 100.0f
#define WORLD_HEIGHT 100.0f
#define STEP_SIZE 0.1f // Step size for tree growth AND connection attempts
#define START_X 10.0f
#define START_Y 10.0f
#define GOAL_X 90.0f
#define GOAL_Y 90.0f

// RRT parameters
#define MAX_ITERATIONS 10000 // Max kernel launches (iterations)
#define MAX_NODES_PER_TREE 5000000 // Max nodes per tree
#define MAX_TOTAL_NODES (MAX_NODES_PER_TREE * 2)

// Obstacle parameters
#define OBSTACLE_WIDTH (WORLD_WIDTH / 10.0f)
#define OBSTACLE1_X (WORLD_WIDTH / 3.0f - OBSTACLE_WIDTH / 2.0f)
#define OBSTACLE1_Y 0.0f
#define OBSTACLE1_HEIGHT (0.6f * WORLD_HEIGHT)
#define OBSTACLE2_X (2.0f * WORLD_WIDTH / 3.0f - OBSTACLE_WIDTH / 2.0f)
#define OBSTACLE2_HEIGHT (0.6f * WORLD_HEIGHT)
#define OBSTACLE2_Y (WORLD_HEIGHT - OBSTACLE2_HEIGHT)

// CUDA execution parameters
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 64
// We'll also use shared memory
#define SHARED_MEM_SIZE (THREADS_PER_BLOCK * (sizeof(float) + sizeof(int)))

// --- Data Structures ---
// Node structure remains the same
typedef struct {
    float x;
    float y;
    int parent_idx; // Index of the parent node (-1 for root nodes)
} Node;

// Obstacle structure remains the same
typedef struct {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} Obstacle;

// --- Tree Indices ---
// Define indices for the two trees
#define START_TREE_IDX 0
#define GOAL_TREE_IDX 1

// --- Device Helper Functions ---

// Calculate squared Euclidean distance
__device__ inline float distance_sq(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

// Clamp coordinates to world boundaries
__device__ inline float clamp(float val, float min_val, float max_val) {
    return fminf(max_val, fmaxf(min_val, val));
}

// Collision Check: Line segment vs AABB obstacles
__device__ bool is_collision(float x1, float y1, float x2, float y2,
                             const Obstacle* obstacles, int num_obstacles)
{
    // Endpoint check
    for (int i = 0; i < num_obstacles; ++i) {
        if (x2 >= obstacles[i].x_min && x2 <= obstacles[i].x_max &&
            y2 >= obstacles[i].y_min && y2 <= obstacles[i].y_max) {
            return true;
        }
    }
    
    // Midpoint check (Simplified)
    float mid_x = (x1 + x2) / 2.0f;
    float mid_y = (y1 + y2) / 2.0f;
    for (int i = 0; i < num_obstacles; ++i) {
        if (mid_x >= obstacles[i].x_min && mid_x <= obstacles[i].x_max &&
            mid_y >= obstacles[i].y_min && mid_y <= obstacles[i].y_max) {
            return true;
        }
    }
    return false;
}

// Cooperative parallel nearest neighbor search
__device__ void find_nearest_parallel(
    Node* d_nodes,
    int node_offset,
    int node_count,
    float query_x, 
    float query_y,
    int* d_min_idx,
    float* d_min_dist_sq,
    float* s_dists,
    int* s_indices
) {
    int tid = threadIdx.x;
    
    // Each thread starts with worst-case values
    float min_dist_sq = HUGE_VALF;
    int min_idx = -1;
    
    // Divide nodes among threads
    int nodes_per_thread = (node_count + blockDim.x - 1) / blockDim.x;
    int start_idx = tid * nodes_per_thread;
    int end_idx = min(start_idx + nodes_per_thread, node_count);
    
    // Each thread finds minimum in its segment
    for (int i = start_idx; i < end_idx; i++) {
        int node_idx = node_offset + i;
        float dist_sq = distance_sq(d_nodes[node_idx].x, d_nodes[node_idx].y, query_x, query_y);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_idx = i; // Store local index within tree
        }
    }
    
    // Store thread results in shared memory
    s_dists[tid] = min_dist_sq;
    s_indices[tid] = min_idx;
    __syncthreads();
    
    // Parallel reduction to find global minimum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_dists[tid] > s_dists[tid + stride]) {
                s_dists[tid] = s_dists[tid + stride];
                s_indices[tid] = s_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 has final result
    if (tid == 0) {
        *d_min_dist_sq = s_dists[0];
        *d_min_idx = (s_indices[0] >= 0) ? (node_offset + s_indices[0]) : -1;
    }
    __syncthreads(); // Make sure all threads see the final result
}

// --- CUDA Kernels ---

// Kernel to initialize cuRAND states
__global__ void initialize_rng(curandState *states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Optimized Bi-directional RRT Kernel
__global__ void optimized_birrt_kernel(
    Node* d_nodes,                  // Combined array for both trees
    int* d_node_count_start,        // Node count for start tree
    int* d_node_count_goal,         // Node count for goal tree
    const Obstacle* d_obstacles,    // Array of obstacles 
    int num_obstacles,              // Number of obstacles
    curandState* rng_states,        // RNG states
    int* d_connection_made,         // Flag: 0 = no connection, 1 = connection found
    int* d_connection_node_idx_start, // Index of connection node in start tree
    int* d_connection_node_idx_goal   // Index of connection node in goal tree
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid; // Global thread ID
    
    // Shared memory for parallel reductions
    extern __shared__ float s_mem[];
    float* s_dists = s_mem;
    int* s_indices = (int*)&s_dists[blockDim.x];
    
    // Shared variables for coordination within block
    __shared__ bool active_block;
    __shared__ int target_tree;
    __shared__ int node_offset_target;
    __shared__ int node_offset_other;
    __shared__ int current_node_count_target;
    __shared__ int current_node_count_other;
    __shared__ float q_rand_x;
    __shared__ float q_rand_y;
    __shared__ float q_near_x;
    __shared__ float q_near_y;
    __shared__ float q_new_x;
    __shared__ float q_new_y;
    __shared__ int nearest_node_global_idx;
    __shared__ float min_dist_sq;
    __shared__ int new_node_global_idx;
    __shared__ bool collision_free;
    
    // Load RNG state
    curandState local_rng_state = rng_states[gid];
    
    // Step 1: Check if connection already found
    if (tid == 0) {
        active_block = (*d_connection_made == 0);
    }
    __syncthreads();
    
    if (!active_block) {
        // Save RNG state and exit early
        rng_states[gid] = local_rng_state;
        return;
    }
    
    // Step 2: Thread 0 determines which tree this block works on
    if (tid == 0) {
        // Even blocks work on start tree, odd on goal tree
        target_tree = bid % 2;
        
        if (target_tree == START_TREE_IDX) { // Start tree
            node_offset_target = 0;
            node_offset_other = MAX_NODES_PER_TREE;
            current_node_count_target = *d_node_count_start;
            current_node_count_other = *d_node_count_goal;
        } else { // Goal tree
            node_offset_target = MAX_NODES_PER_TREE;
            node_offset_other = 0;
            current_node_count_target = *d_node_count_goal;
            current_node_count_other = *d_node_count_start;
        }
        
        // Check if the trees are valid
        active_block = (current_node_count_target > 0 && 
                       current_node_count_target < MAX_NODES_PER_TREE &&
                       current_node_count_other > 0);
    }
    __syncthreads();
    
    if (!active_block) {
        // Save RNG state and exit early
        rng_states[gid] = local_rng_state;
        return;
    }
    
    // Step 3: Thread 0 samples q_rand
    if (tid == 0) {
        q_rand_x = curand_uniform(&local_rng_state) * WORLD_WIDTH;
        q_rand_y = curand_uniform(&local_rng_state) * WORLD_HEIGHT;
    }
    __syncthreads();
    
    // Step 4: All threads cooperate to find nearest neighbor in target tree
    find_nearest_parallel(
        d_nodes, 
        node_offset_target, 
        current_node_count_target, 
        q_rand_x, 
        q_rand_y, 
        &nearest_node_global_idx, 
        &min_dist_sq,
        s_dists,
        s_indices
    );
    
    // Check if we found a valid nearest neighbor
    if (tid == 0) {
        if (nearest_node_global_idx == -1) {
            active_block = false;
        } else {
            // Get the nearest node coordinates for steering
            q_near_x = d_nodes[nearest_node_global_idx].x;
            q_near_y = d_nodes[nearest_node_global_idx].y;
        }
    }
    __syncthreads();
    
    if (!active_block) {
        // Save RNG state and exit early
        rng_states[gid] = local_rng_state;
        return;
    }
    
    // Step 5: Thread 0 steers from q_near towards q_rand to get q_new
    if (tid == 0) {
        float dir_x = q_rand_x - q_near_x;
        float dir_y = q_rand_y - q_near_y;
        float mag_sq = dir_x * dir_x + dir_y * dir_y;
        
        if (mag_sq <= STEP_SIZE * STEP_SIZE || mag_sq == 0.0f) {
            q_new_x = q_rand_x;
            q_new_y = q_rand_y;
        } else {
            float mag = sqrtf(mag_sq);
            float scale = STEP_SIZE / mag;
            q_new_x = q_near_x + dir_x * scale;
            q_new_y = q_near_y + dir_y * scale;
        }
        
        // Clamp to world boundaries
        q_new_x = clamp(q_new_x, 0.0f, WORLD_WIDTH);
        q_new_y = clamp(q_new_y, 0.0f, WORLD_HEIGHT);
    }
    __syncthreads();
    
    // Step 6: Thread 0 does collision check
    if (tid == 0) {
        collision_free = !is_collision(q_near_x, q_near_y, q_new_x, q_new_y, d_obstacles, num_obstacles);
        active_block = collision_free;
    }
    __syncthreads();
    
    if (!active_block) {
        // Save RNG state and exit early
        rng_states[gid] = local_rng_state;
        return;
    }
    
    // Step 7: Thread 0 adds node to target tree
    if (tid == 0) {
        int *d_node_count_target;
        if (target_tree == START_TREE_IDX) {
            d_node_count_target = d_node_count_start;
        } else {
            d_node_count_target = d_node_count_goal;
        }
        
        // Atomically get index for new node
        int new_node_local_idx = atomicAdd(d_node_count_target, 1);
        
        // Check if tree is full
        if (new_node_local_idx >= MAX_NODES_PER_TREE) {
            atomicSub(d_node_count_target, 1);
            active_block = false;
        } else {
            // Add new node
            new_node_global_idx = node_offset_target + new_node_local_idx;
            d_nodes[new_node_global_idx].x = q_new_x;
            d_nodes[new_node_global_idx].y = q_new_y;
            d_nodes[new_node_global_idx].parent_idx = nearest_node_global_idx;
        }
    }
    __syncthreads();
    
    if (!active_block) {
        // Save RNG state and exit early
        rng_states[gid] = local_rng_state;
        return;
    }
    
    // Step 8: All threads cooperate to find nearest neighbor in other tree for connection attempt
    // Set up shared variables for nearest neighbor search
    __shared__ int connect_node_global_idx;
    __shared__ float connect_min_dist_sq;
    
    find_nearest_parallel(
        d_nodes, 
        node_offset_other, 
        current_node_count_other, 
        q_new_x, 
        q_new_y, 
        &connect_node_global_idx,
        &connect_min_dist_sq,
        s_dists,
        s_indices
    );
    
    // Thread 0 checks if connection can be made
    if (tid == 0 && connect_node_global_idx != -1) {
        // Check if distance is within step size
        if (connect_min_dist_sq <= STEP_SIZE * STEP_SIZE) {
            float q_connect_x = d_nodes[connect_node_global_idx].x;
            float q_connect_y = d_nodes[connect_node_global_idx].y;
            
            // Check if connection path is collision-free
            if (!is_collision(q_new_x, q_new_y, q_connect_x, q_connect_y, d_obstacles, num_obstacles)) {
                // Attempt to be the first to make connection
                if (atomicExch(d_connection_made, 1) == 0) {
                    // This block is the first to make a connection
                    if (target_tree == START_TREE_IDX) {
                        atomicExch(d_connection_node_idx_start, new_node_global_idx);
                        atomicExch(d_connection_node_idx_goal, connect_node_global_idx);
                    } else {
                        atomicExch(d_connection_node_idx_start, connect_node_global_idx);
                        atomicExch(d_connection_node_idx_goal, new_node_global_idx);
                    }
                }
            }
        }
    }
    
    // Save RNG state
    rng_states[gid] = local_rng_state;
}

int main() {
    // --- Setup Obstacles ---
    std::vector<Obstacle> h_obstacles;
    h_obstacles.push_back({OBSTACLE1_X, OBSTACLE1_Y, OBSTACLE1_X + OBSTACLE_WIDTH, OBSTACLE1_Y + OBSTACLE1_HEIGHT});
    h_obstacles.push_back({OBSTACLE2_X, OBSTACLE2_Y, OBSTACLE2_X + OBSTACLE_WIDTH, OBSTACLE2_Y + OBSTACLE2_HEIGHT});
    int num_obstacles = h_obstacles.size();

    // --- Print Simulation Parameters ---
    printf("--- Optimized Bi-directional RRT Simulation Parameters ---\n");
    printf("World:       %.1f x %.1f\n", WORLD_WIDTH, WORLD_HEIGHT);
    printf("Start:       (%.1f, %.1f)\n", START_X, START_Y);
    printf("Goal:        (%.1f, %.1f)\n", GOAL_X, GOAL_Y);
    printf("Step Size:   %.2f\n", STEP_SIZE);
    printf("Obstacles:   %d\n", num_obstacles);
    for(int i=0; i<num_obstacles; ++i) {
        printf("  Obs %d:    (%.2f, %.2f) to (%.2f, %.2f)\n", i+1,
              h_obstacles[i].x_min, h_obstacles[i].y_min, h_obstacles[i].x_max, h_obstacles[i].y_max);
    }
    printf("Max Nodes/Tree: %d (Total: %d)\n", MAX_NODES_PER_TREE, MAX_TOTAL_NODES);
    printf("Max Iters:   %d\n", MAX_ITERATIONS);
    printf("--- CUDA Parameters ---\n");
    printf("Threads/Blk: %d\n", THREADS_PER_BLOCK);
    printf("Blocks:      %d\n", NUM_BLOCKS);
    printf("Total Thrds: %d\n", NUM_BLOCKS * THREADS_PER_BLOCK);
    printf("---------------------------------\n");

    // --- Host Memory Allocation ---
    Node* h_nodes = (Node*)malloc(MAX_TOTAL_NODES * sizeof(Node));
    if (!h_nodes) { fprintf(stderr, "Failed to allocate host memory for nodes.\n"); return 1; }

    // --- Device Memory Allocation ---
    Node* d_nodes;
    Obstacle* d_obstacles;
    curandState* d_rng_states;
    int* d_node_count_start;
    int* d_node_count_goal;
    int* d_connection_made;
    int* d_connection_node_idx_start;
    int* d_connection_node_idx_goal;

    CUDA_CHECK(cudaMalloc(&d_nodes, MAX_TOTAL_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_obstacles, num_obstacles * sizeof(Obstacle)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_node_count_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_count_goal, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_made, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_node_idx_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_node_idx_goal, sizeof(int)));

    // --- Initialization ---
    printf("Initializing...\n");

    // Initialize start node (at index 0)
    h_nodes[0].x = START_X;
    h_nodes[0].y = START_Y;
    h_nodes[0].parent_idx = -1; // Root of start tree
    int h_node_count_start = 1;

    // Initialize goal node (at index MAX_NODES_PER_TREE)
    int goal_node_start_index = MAX_NODES_PER_TREE;
    h_nodes[goal_node_start_index].x = GOAL_X;
    h_nodes[goal_node_start_index].y = GOAL_Y;
    h_nodes[goal_node_start_index].parent_idx = -1; // Root of goal tree
    int h_node_count_goal = 1;

    // Initialize connection status
    int h_connection_made = 0;
    int h_connection_node_idx_start = -1;
    int h_connection_node_idx_goal = -1;

    // Copy initial data to device
    printf("  Copying initial data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_nodes, &h_nodes[0], sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes + goal_node_start_index, &h_nodes[goal_node_start_index], sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count_start, &h_node_count_start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count_goal, &h_node_count_goal, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connection_made, &h_connection_made, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connection_node_idx_start, &h_connection_node_idx_start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connection_node_idx_goal, &h_connection_node_idx_goal, sizeof(int), cudaMemcpyHostToDevice));

    // Initialize RNG states
    printf("  Initializing RNG states...\n");
    initialize_rng<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_rng_states, time(0));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Initialization complete.\n");
    printf("Starting Optimized Bi-directional RRT...\n");
    
    // --- Create CUDA events for timing ---
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    // Record start time
    CUDA_CHECK(cudaEventRecord(start_event));
    auto cpu_start_time = std::chrono::high_resolution_clock::now();

    // --- Main Bi-RRT Loop ---
    int iteration;
    for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        // Launch the optimized kernel
        optimized_birrt_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM_SIZE>>>(
            d_nodes, d_node_count_start, d_node_count_goal,
            d_obstacles, num_obstacles, d_rng_states,
            d_connection_made, d_connection_node_idx_start, d_connection_node_idx_goal);
        
        CUDA_CHECK(cudaGetLastError());
        
        // Periodically check if a connection has been made
        const int check_interval = 20; // Check more frequently due to more blocks working in parallel
        if ((iteration + 1) % check_interval == 0 || iteration == MAX_ITERATIONS - 1) {
            CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel
            
            CUDA_CHECK(cudaMemcpy(&h_connection_made, d_connection_made, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_connection_made == 1) {
                CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
                printf("\nConnection established after %d iterations! Nodes: Start=%d, Goal=%d\n",
                       iteration + 1, h_node_count_start, h_node_count_goal);
                break; // Exit loop
            }
            
            // Optional: Print progress
            const int print_interval = 500;
            if ((iteration + 1) % print_interval == 0) {
                CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
                printf("Iteration %d, Nodes: Start=%d, Goal=%d\n",
                      iteration + 1, h_node_count_start, h_node_count_goal);
            }
            
            // Check if either tree is full
            if ((iteration + 1) % print_interval != 0) {
                CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
            }
            if (h_node_count_start >= MAX_NODES_PER_TREE || h_node_count_goal >= MAX_NODES_PER_TREE) {
                printf("\nMaximum nodes per tree (%d) reached.\n", MAX_NODES_PER_TREE);
                break; // Exit loop
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // Final sync
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    
    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event));
    std::chrono::duration<double, std::milli> cpu_duration_ms = cpu_end_time - cpu_start_time;
    
    printf("Optimized Bi-RRT computation finished in %.3f ms (GPU time) / %.3f ms (CPU time).\n", 
           gpu_time_ms, cpu_duration_ms.count());

    // --- Path Reconstruction (if connection made) ---
    CUDA_CHECK(cudaMemcpy(&h_connection_made, d_connection_made, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> path_indices; // Store final path indices

    if (h_connection_made) {
        // Get final counts and connection indices
        CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_connection_node_idx_start, d_connection_node_idx_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_connection_node_idx_goal, d_connection_node_idx_goal, sizeof(int), cudaMemcpyDeviceToHost));

        printf("Reconstructing path from connection nodes: Start Tree Idx=%d, Goal Tree Idx=%d\n",
               h_connection_node_idx_start, h_connection_node_idx_goal);
        printf("Total Nodes: Start=%d, Goal=%d\n", h_node_count_start, h_node_count_goal);

        // Copy necessary nodes back from device
        printf("  Copying nodes from device...\n");
        CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, h_node_count_start * sizeof(Node), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nodes + MAX_NODES_PER_TREE,
                              d_nodes + MAX_NODES_PER_TREE,
                              h_node_count_goal * sizeof(Node),
                              cudaMemcpyDeviceToHost));

        // Reconstruct path from start tree
        std::vector<int> path_start_segment;
        int current_idx = h_connection_node_idx_start;
        int path_steps = 0;
        const int MAX_PATH_STEPS = h_node_count_start + h_node_count_goal + 2; // Safety break

        printf("  Tracing start tree segment...\n");
        while (current_idx != -1 && path_steps < MAX_PATH_STEPS) {
            if (current_idx < 0 || current_idx >= MAX_NODES_PER_TREE) {
                fprintf(stderr, "Error: Invalid node index %d in start tree during path reconstruction.\n", current_idx);
                path_start_segment.clear(); break;
            }
            path_start_segment.push_back(current_idx);
            current_idx = h_nodes[current_idx].parent_idx;
            path_steps++;
        }
        if (current_idx != -1 || path_steps >= MAX_PATH_STEPS) {
            fprintf(stderr, "Error: Start path reconstruction failed or exceeded max steps.\n");
            path_start_segment.clear();
        } else {
            std::reverse(path_start_segment.begin(), path_start_segment.end());
        }

        // Reconstruct path from goal tree
        std::vector<int> path_goal_segment;
        current_idx = h_connection_node_idx_goal;
        path_steps = 0;

        printf("  Tracing goal tree segment...\n");
        while (current_idx != -1 && path_steps < MAX_PATH_STEPS) {
            if (current_idx < MAX_NODES_PER_TREE || current_idx >= MAX_TOTAL_NODES) {
                fprintf(stderr, "Error: Invalid node index %d in goal tree during path reconstruction.\n", current_idx);
                path_goal_segment.clear(); break;
            }
            path_goal_segment.push_back(current_idx);
            current_idx = h_nodes[current_idx].parent_idx;
            path_steps++;
        }
        if (current_idx != -1 || path_steps >= MAX_PATH_STEPS) {
            fprintf(stderr, "Error: Goal path reconstruction failed or exceeded max steps.\n");
            path_goal_segment.clear();
        }

        // Combine paths
        if (!path_start_segment.empty() && !path_goal_segment.empty()) {
            path_indices = path_start_segment;
            path_indices.insert(path_indices.end(), path_goal_segment.begin(), path_goal_segment.end());
            printf("Path Found (%d steps).\n", (int)path_indices.size());
        } else {
            printf("Path reconstruction failed (one or both segments invalid).\n");
            h_connection_made = 0;
        }

    } else {
        // Connection not made
        CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
        if (iteration >= MAX_ITERATIONS) {
            printf("Connection not established within %d iterations. Nodes: Start=%d, Goal=%d\n",
                   MAX_ITERATIONS, h_node_count_start, h_node_count_goal);
        } else if (h_node_count_start >= MAX_NODES_PER_TREE || h_node_count_goal >= MAX_NODES_PER_TREE){
            printf("Connection not established. Tree reached maximum size. Nodes: Start=%d, Goal=%d\n",
                   h_node_count_start, h_node_count_goal);
        } else {
            printf("Connection not established. Status: iterations=%d, Nodes: Start=%d, Goal=%d\n",
                   iteration, h_node_count_start, h_node_count_goal);
        }
    }

    // --- Save results to CSV ---
    if(h_node_count_start <= 1 || h_node_count_goal <= 1) {
        CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    // Copy nodes if connection wasn't made
    if (!h_connection_made) {
        printf("Copying nodes from device for saving trees...\n");
        if(h_node_count_start > 0)
            CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, h_node_count_start * sizeof(Node), cudaMemcpyDeviceToHost));
        if(h_node_count_goal > 0)
            CUDA_CHECK(cudaMemcpy(h_nodes + MAX_NODES_PER_TREE, d_nodes + MAX_NODES_PER_TREE, h_node_count_goal * sizeof(Node), cudaMemcpyDeviceToHost));
    }

    printf("Saving results to CSV files...\n");

    // Save node data
    FILE* nodes_file = fopen("birrt_nodes_optimized.csv", "w");
    if (nodes_file == NULL) {
        fprintf(stderr, "Error opening nodes output file 'birrt_nodes_optimized.csv'\n");
    } else {
        fprintf(nodes_file, "global_id,x,y,parent_global_id,tree_type,on_path\n");
        std::vector<bool> on_path(MAX_TOTAL_NODES, false);
        if (h_connection_made && !path_indices.empty()) {
            for (int idx : path_indices) {
                if (idx >= 0 && idx < MAX_TOTAL_NODES) {
                    on_path[idx] = true;
                }
            }
        }

        // Write start tree nodes
        for (int i = 0; i < h_node_count_start; ++i) {
            int global_idx = i;
            fprintf(nodes_file, "%d,%.4f,%.4f,%d,%d,%d\n",
                   global_idx, h_nodes[global_idx].x, h_nodes[global_idx].y,
                   h_nodes[global_idx].parent_idx,
                   START_TREE_IDX,
                   on_path[global_idx] ? 1 : 0);
        }
        
        // Write goal tree nodes
        for (int i = 0; i < h_node_count_goal; ++i) {
            int global_idx = MAX_NODES_PER_TREE + i;
            fprintf(nodes_file, "%d,%.4f,%.4f,%d,%d,%d\n",
                   global_idx, h_nodes[global_idx].x, h_nodes[global_idx].y,
                   h_nodes[global_idx].parent_idx,
                   GOAL_TREE_IDX,
                   on_path[global_idx] ? 1 : 0);
        }
        fclose(nodes_file);
        printf("Saved %d start tree nodes and %d goal tree nodes to birrt_nodes_optimized.csv\n",
               h_node_count_start, h_node_count_goal);
    }

    // Save world data
    FILE* world_file = fopen("birrt_world_optimized.csv", "w");
    if (world_file == NULL) {
        fprintf(stderr, "Error opening world output file 'birrt_world_optimized.csv'\n");
    } else {
        fprintf(world_file, "type,x,y,width,height_or_threshold\n");
        fprintf(world_file, "world,0,0,%.2f,%.2f\n", WORLD_WIDTH, WORLD_HEIGHT);
        fprintf(world_file, "start,%.2f,%.2f,0,0\n", START_X, START_Y);
        fprintf(world_file, "goal,%.2f,%.2f,0,0\n", GOAL_X, GOAL_Y);
        for (size_t i = 0; i < h_obstacles.size(); i++) {
            fprintf(world_file, "obstacle,%.2f,%.2f,%.2f,%.2f\n",
                   h_obstacles[i].x_min, h_obstacles[i].y_min,
                   h_obstacles[i].x_max - h_obstacles[i].x_min,
                   h_obstacles[i].y_max - h_obstacles[i].y_min);
        }
        fclose(world_file);
        printf("Saved world data to birrt_world_optimized.csv\n");
    }

    // Save performance data
    FILE* perf_file = fopen("birrt_optimized_performance.csv", "w");
    if (perf_file == NULL) {
        fprintf(stderr, "Error opening performance output file 'birrt_optimized_performance.csv'\n");
    } else {
        fprintf(perf_file, "metric,value\n");
        fprintf(perf_file, "total_gpu_time_ms,%.6f\n", gpu_time_ms);
        fprintf(perf_file, "total_cpu_time_ms,%.6f\n", cpu_duration_ms.count());
        fprintf(perf_file, "iterations,%d\n", iteration + 1);
        fprintf(perf_file, "start_tree_nodes,%d\n", h_node_count_start);
        fprintf(perf_file, "goal_tree_nodes,%d\n", h_node_count_goal);
        fprintf(perf_file, "connection_made,%d\n", h_connection_made);
        fprintf(perf_file, "path_length,%d\n", (int)path_indices.size());
        fprintf(perf_file, "blocks,%d\n", NUM_BLOCKS);
        fprintf(perf_file, "threads_per_block,%d\n", THREADS_PER_BLOCK);
        fclose(perf_file);
        printf("Saved performance data to birrt_optimized_performance.csv\n");
    }

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    free(h_nodes);
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_obstacles));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_node_count_start));
    CUDA_CHECK(cudaFree(d_node_count_goal));
    CUDA_CHECK(cudaFree(d_connection_made));
    CUDA_CHECK(cudaFree(d_connection_node_idx_start));
    CUDA_CHECK(cudaFree(d_connection_node_idx_goal));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    printf("Done.\n");
    return 0;
}