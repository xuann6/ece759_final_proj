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
// Goal threshold not directly used in Bi-RRT, connection is the goal

// RRT parameters
// Goal bias less relevant, trees grow towards each other
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
#define THREADS_PER_BLOCK 128
// Use sufficient blocks for parallelism
#define NUM_BLOCKS 1

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

// --- Statistics for Timing ---
struct StepTiming {
    float step1_check_connection;
    float step2_determine_tree;
    float step3_sample_q_rand;
    float step4_find_nearest;
    float step5_steer;
    float step6_collision_check;
    float step7_add_node;
    float step8_attempt_connection;
};

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

// Collision Check: Line segment vs AABB obstacles (Same as original)
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


// --- CUDA Kernels ---

// Kernel to initialize cuRAND states (Same as original)
__global__ void initialize_rng(curandState *states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// For timing, we'll split the birrt_iteration_kernel into separate steps
// These kernels will share state via device memory

// Step 1: Check if connection already found
__global__ void step1_check_connection(
    int* d_connection_made,
    int* d_thread_active,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Thread is active by default
    d_thread_active[tid] = 1;
    
    // If connection already made, deactivate thread
    if (*d_connection_made == 1) {
        d_thread_active[tid] = 0;
    }
    
    rng_states[tid] = local_rng_state;
}

// Step 2: Determine which tree this thread works on
__global__ void step2_determine_tree(
    int* d_thread_active,
    int* d_node_count_start,
    int* d_node_count_goal,
    int* d_target_tree,
    int* d_node_offset_target,
    int* d_node_offset_other,
    int* d_current_node_count_target,
    int* d_current_node_count_other,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    // Simple alternation: even threads work on start tree, odd on goal tree
    int target_tree = tid % 2; // 0 for start tree, 1 for goal tree
    d_target_tree[tid] = target_tree;
    
    int node_offset_target;
    int node_offset_other;
    int current_node_count_target;
    int current_node_count_other;
    
    if (target_tree == START_TREE_IDX) { // Grow Start Tree
        node_offset_target = 0;
        node_offset_other = MAX_NODES_PER_TREE; // Goal tree starts at midpoint
        current_node_count_target = *d_node_count_start;
        current_node_count_other = *d_node_count_goal;
    } else { // Grow Goal Tree
        node_offset_target = MAX_NODES_PER_TREE; // Goal tree starts at midpoint
        node_offset_other = 0;
        current_node_count_target = *d_node_count_goal;
        current_node_count_other = *d_node_count_start;
    }
    
    d_node_offset_target[tid] = node_offset_target;
    d_node_offset_other[tid] = node_offset_other;
    d_current_node_count_target[tid] = current_node_count_target;
    d_current_node_count_other[tid] = current_node_count_other;
    
    // Check if target tree is valid and not full
    if (current_node_count_target <= 0 || current_node_count_target >= MAX_NODES_PER_TREE) {
        d_thread_active[tid] = 0;
    }
    // Check if other tree has at least one node (needed for connection attempts)
    if (current_node_count_other <= 0) {
        d_thread_active[tid] = 0;
    }
    
    rng_states[tid] = local_rng_state;
}

// Step 3: Sample q_rand
__global__ void step3_sample_q_rand(
    int* d_thread_active,
    float* d_q_rand_x,
    float* d_q_rand_y,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    // Simple uniform sampling for Bi-RRT
    d_q_rand_x[tid] = curand_uniform(&local_rng_state) * WORLD_WIDTH;
    d_q_rand_y[tid] = curand_uniform(&local_rng_state) * WORLD_HEIGHT;
    
    rng_states[tid] = local_rng_state;
}

// Step 4: Find Nearest Neighbor in the TARGET tree
__global__ void step4_find_nearest(
    int* d_thread_active,
    Node* d_nodes,
    int* d_target_tree,
    int* d_node_offset_target,
    int* d_current_node_count_target,
    float* d_q_rand_x,
    float* d_q_rand_y,
    int* d_nearest_node_local_idx,
    int* d_nearest_node_global_idx,
    float* d_q_near_x,
    float* d_q_near_y,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    int node_offset_target = d_node_offset_target[tid];
    int current_node_count_target = d_current_node_count_target[tid];
    float q_rand_x = d_q_rand_x[tid];
    float q_rand_y = d_q_rand_y[tid];
    
    // Find nearest node in the target tree
    int nearest_node_local_idx = -1;
    float min_dist_sq = HUGE_VALF;
    
    // Linear scan through existing nodes in the target tree
    for (int i = 0; i < current_node_count_target; ++i) {
        int node_global_idx = node_offset_target + i; // Calculate global index
        float dist_sq = distance_sq(d_nodes[node_global_idx].x, d_nodes[node_global_idx].y, q_rand_x, q_rand_y);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_node_local_idx = i; // Store local index
        }
    }
    
    // Should always find a nearest node if count > 0
    if (nearest_node_local_idx == -1) {
        d_thread_active[tid] = 0;
        rng_states[tid] = local_rng_state;
        return;
    }
    
    int nearest_node_global_idx = node_offset_target + nearest_node_local_idx;
    d_nearest_node_local_idx[tid] = nearest_node_local_idx;
    d_nearest_node_global_idx[tid] = nearest_node_global_idx;
    d_q_near_x[tid] = d_nodes[nearest_node_global_idx].x;
    d_q_near_y[tid] = d_nodes[nearest_node_global_idx].y;
    
    rng_states[tid] = local_rng_state;
}

// Step 5: Steer from q_near towards q_rand to get q_new
__global__ void step5_steer(
    int* d_thread_active,
    float* d_q_near_x,
    float* d_q_near_y,
    float* d_q_rand_x,
    float* d_q_rand_y,
    float* d_q_new_x,
    float* d_q_new_y,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    float q_near_x = d_q_near_x[tid];
    float q_near_y = d_q_near_y[tid];
    float q_rand_x = d_q_rand_x[tid];
    float q_rand_y = d_q_rand_y[tid];
    
    // Steer from q_near towards q_rand to get q_new
    float dir_x = q_rand_x - q_near_x;
    float dir_y = q_rand_y - q_near_y;
    float mag_sq = dir_x * dir_x + dir_y * dir_y; // Squared magnitude
    float q_new_x, q_new_y;
    
    if (mag_sq <= STEP_SIZE * STEP_SIZE || mag_sq == 0.0f) { // If q_rand is closer than step_size or at same point
        q_new_x = q_rand_x;
        q_new_y = q_rand_y;
    } else { // Steer by STEP_SIZE
        float mag = sqrtf(mag_sq);
        float scale = STEP_SIZE / mag;
        q_new_x = q_near_x + dir_x * scale;
        q_new_y = q_near_y + dir_y * scale;
    }
    
    // Clamp q_new to world boundaries
    q_new_x = clamp(q_new_x, 0.0f, WORLD_WIDTH);
    q_new_y = clamp(q_new_y, 0.0f, WORLD_HEIGHT);
    
    d_q_new_x[tid] = q_new_x;
    d_q_new_y[tid] = q_new_y;
    
    rng_states[tid] = local_rng_state;
}

// Step 6: Collision Check for the new segment
__global__ void step6_collision_check(
    int* d_thread_active,
    float* d_q_near_x,
    float* d_q_near_y,
    float* d_q_new_x,
    float* d_q_new_y,
    Obstacle* d_obstacles,
    int num_obstacles,
    int* d_collision_free,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    float q_near_x = d_q_near_x[tid];
    float q_near_y = d_q_near_y[tid];
    float q_new_x = d_q_new_x[tid];
    float q_new_y = d_q_new_y[tid];
    
    // Collision Check
    bool collision = is_collision(q_near_x, q_near_y, q_new_x, q_new_y, d_obstacles, num_obstacles);
    d_collision_free[tid] = collision ? 0 : 1;
    
    // If collision, deactivate thread
    if (collision) {
        d_thread_active[tid] = 0;
    }
    
    rng_states[tid] = local_rng_state;
}

// Step 7: Add Node to TARGET Tree
__global__ void step7_add_node(
    int* d_thread_active,
    Node* d_nodes,
    int* d_target_tree,
    int* d_node_count_start,
    int* d_node_count_goal,
    int* d_node_offset_target,
    float* d_q_new_x,
    float* d_q_new_y,
    int* d_nearest_node_global_idx,
    int* d_new_node_local_idx,
    int* d_new_node_global_idx,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    int target_tree = d_target_tree[tid];
    int node_offset_target = d_node_offset_target[tid];
    int* d_node_count_target;
    
    if (target_tree == START_TREE_IDX) {
        d_node_count_target = d_node_count_start;
    } else {
        d_node_count_target = d_node_count_goal;
    }
    
    // Atomically get the local index for the new node within its tree
    int new_node_local_idx = atomicAdd(d_node_count_target, 1);
    d_new_node_local_idx[tid] = new_node_local_idx;
    
    // Check if the target tree is full BEFORE writing node data
    if (new_node_local_idx >= MAX_NODES_PER_TREE) {
        atomicSub(d_node_count_target, 1);
        d_thread_active[tid] = 0;
        rng_states[tid] = local_rng_state;
        return;
    }
    
    int new_node_global_idx = node_offset_target + new_node_local_idx;
    d_new_node_global_idx[tid] = new_node_global_idx;
    
    // Write the new node data
    d_nodes[new_node_global_idx].x = d_q_new_x[tid];
    d_nodes[new_node_global_idx].y = d_q_new_y[tid];
    // Parent index is the GLOBAL index of the nearest node found earlier
    d_nodes[new_node_global_idx].parent_idx = d_nearest_node_global_idx[tid];
    
    rng_states[tid] = local_rng_state;
}

// Step 8: Attempt Connection to the OTHER tree
__global__ void step8_attempt_connection(
    int* d_thread_active,
    Node* d_nodes,
    int* d_target_tree, 
    int* d_node_offset_other,
    int* d_current_node_count_other,
    float* d_q_new_x,
    float* d_q_new_y,
    int* d_new_node_global_idx,
    Obstacle* d_obstacles,
    int num_obstacles,
    int* d_connection_made,
    int* d_connection_node_idx_start,
    int* d_connection_node_idx_goal,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng_state = rng_states[tid];
    
    // Skip if thread is not active
    if (d_thread_active[tid] == 0) {
        rng_states[tid] = local_rng_state;
        return;
    }
    
    int target_tree = d_target_tree[tid];
    int node_offset_other = d_node_offset_other[tid];
    int current_node_count_other = d_current_node_count_other[tid];
    float q_new_x = d_q_new_x[tid];
    float q_new_y = d_q_new_y[tid];
    int new_node_global_idx = d_new_node_global_idx[tid];
    
    // Find nearest node in the OTHER tree to the new node
    int connect_node_other_local_idx = -1;
    float min_dist_sq_other = HUGE_VALF;
    
    // Linear scan through the OTHER tree
    for (int i = 0; i < current_node_count_other; ++i) {
        int node_other_global_idx = node_offset_other + i;
        float dist_sq = distance_sq(d_nodes[node_other_global_idx].x, d_nodes[node_other_global_idx].y, q_new_x, q_new_y);
        if (dist_sq < min_dist_sq_other) {
            min_dist_sq_other = dist_sq;
            connect_node_other_local_idx = i; // Store local index relative to other tree
        }
    }
    
    // If a nearest node was found in the other tree
    if (connect_node_other_local_idx != -1) {
        int connect_node_other_global_idx = node_offset_other + connect_node_other_local_idx;
        float q_connect_x = d_nodes[connect_node_other_global_idx].x;
        float q_connect_y = d_nodes[connect_node_other_global_idx].y;
        
        // Check if q_new and q_connect are close enough (within STEP_SIZE)
        // AND the direct path between them is collision-free
        if (min_dist_sq_other <= STEP_SIZE * STEP_SIZE) { // Check distance first
            if (!is_collision(q_new_x, q_new_y, q_connect_x, q_connect_y, d_obstacles, num_obstacles)) {
                // --- Connection Successful! ---
                // Atomically set the connection flag and store indices
                if (atomicExch(d_connection_made, 1) == 0) {
                    // This thread is the first to make the connection
                    // Store the GLOBAL indices of the two connecting nodes
                    if (target_tree == START_TREE_IDX) {
                        atomicExch(d_connection_node_idx_start, new_node_global_idx);
                        atomicExch(d_connection_node_idx_goal, connect_node_other_global_idx);
                    } else { // target_tree == GOAL_TREE_IDX
                        atomicExch(d_connection_node_idx_start, connect_node_other_global_idx);
                        atomicExch(d_connection_node_idx_goal, new_node_global_idx);
                    }
                }
            }
        }
    }
    
    rng_states[tid] = local_rng_state;
}

// --- Host Code ---
int main() {
    // --- Setup Obstacles ---
    std::vector<Obstacle> h_obstacles;
    h_obstacles.push_back({OBSTACLE1_X, OBSTACLE1_Y, OBSTACLE1_X + OBSTACLE_WIDTH, OBSTACLE1_Y + OBSTACLE1_HEIGHT});
    h_obstacles.push_back({OBSTACLE2_X, OBSTACLE2_Y, OBSTACLE2_X + OBSTACLE_WIDTH, OBSTACLE2_Y + OBSTACLE2_HEIGHT});
    int num_obstacles = h_obstacles.size();

    // --- Print Simulation Parameters ---
    printf("--- Bi-directional RRT Simulation Parameters ---\n");
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
    int* d_node_count_start; // Count for start tree
    int* d_node_count_goal;  // Count for goal tree
    int* d_connection_made;  // Connection flag
    int* d_connection_node_idx_start; // Connecting node index from start tree
    int* d_connection_node_idx_goal;  // Connecting node index from goal tree
    
    // Temporary arrays for kernel steps
    int* d_thread_active;
    int* d_target_tree;
    int* d_node_offset_target;
    int* d_node_offset_other;
    int* d_current_node_count_target;
    int* d_current_node_count_other;
    float* d_q_rand_x;
    float* d_q_rand_y;
    int* d_nearest_node_local_idx;
    int* d_nearest_node_global_idx;
    float* d_q_near_x;
    float* d_q_near_y;
    float* d_q_new_x;
    float* d_q_new_y;
    int* d_collision_free;
    int* d_new_node_local_idx;
    int* d_new_node_global_idx;
    
    // Allocate main data structures
    CUDA_CHECK(cudaMalloc(&d_nodes, MAX_TOTAL_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_obstacles, num_obstacles * sizeof(Obstacle)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_node_count_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_count_goal, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_made, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_node_idx_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_connection_node_idx_goal, sizeof(int)));
    
    // Allocate temporary arrays
    CUDA_CHECK(cudaMalloc(&d_thread_active, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_target_tree, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_offset_target, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_offset_other, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_current_node_count_target, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_current_node_count_other, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_q_rand_x, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_rand_y, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nearest_node_local_idx, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nearest_node_global_idx, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_q_near_x, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_near_y, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_new_x, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_new_y, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_collision_free, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_node_local_idx, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_node_global_idx, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    
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
    // Copy start node
    CUDA_CHECK(cudaMemcpy(d_nodes, &h_nodes[0], sizeof(Node), cudaMemcpyHostToDevice));
    // Copy goal node
    CUDA_CHECK(cudaMemcpy(d_nodes + goal_node_start_index, &h_nodes[goal_node_start_index], sizeof(Node), cudaMemcpyHostToDevice));
    // Copy obstacles
    CUDA_CHECK(cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice));
    // Copy counts and flags
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
    printf("Starting Bi-directional RRT...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // --- Create CUDA events for timing ---
    cudaEvent_t start_event, stop_event;
    cudaEvent_t step1_start, step1_stop;
    cudaEvent_t step2_start, step2_stop;
    cudaEvent_t step3_start, step3_stop;
    cudaEvent_t step4_start, step4_stop;
    cudaEvent_t step5_start, step5_stop;
    cudaEvent_t step6_start, step6_stop;
    cudaEvent_t step7_start, step7_stop;
    cudaEvent_t step8_start, step8_stop;
    
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&step1_start));
    CUDA_CHECK(cudaEventCreate(&step1_stop));
    CUDA_CHECK(cudaEventCreate(&step2_start));
    CUDA_CHECK(cudaEventCreate(&step2_stop));
    CUDA_CHECK(cudaEventCreate(&step3_start));
    CUDA_CHECK(cudaEventCreate(&step3_stop));
    CUDA_CHECK(cudaEventCreate(&step4_start));
    CUDA_CHECK(cudaEventCreate(&step4_stop));
    CUDA_CHECK(cudaEventCreate(&step5_start));
    CUDA_CHECK(cudaEventCreate(&step5_stop));
    CUDA_CHECK(cudaEventCreate(&step6_start));
    CUDA_CHECK(cudaEventCreate(&step6_stop));
    CUDA_CHECK(cudaEventCreate(&step7_start));
    CUDA_CHECK(cudaEventCreate(&step7_stop));
    CUDA_CHECK(cudaEventCreate(&step8_start));
    CUDA_CHECK(cudaEventCreate(&step8_stop));
    
    // Variables to store cumulative timing results
    StepTiming total_timing = {0};
    StepTiming avg_timing = {0};
    StepTiming min_timing = {HUGE_VALF, HUGE_VALF, HUGE_VALF, HUGE_VALF, HUGE_VALF, HUGE_VALF, HUGE_VALF, HUGE_VALF};
    StepTiming max_timing = {0};
    
    // --- Main Bi-RRT Loop ---
    int iteration;
    for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        // Record step timings for this iteration
        float step1_time, step2_time, step3_time, step4_time;
        float step5_time, step6_time, step7_time, step8_time;
        
        // Step 1: Check if connection already found
        CUDA_CHECK(cudaEventRecord(step1_start));
        step1_check_connection<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_connection_made, d_thread_active, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step1_stop));
        
        // Step 2: Determine which tree to work on
        CUDA_CHECK(cudaEventRecord(step2_start));
        step2_determine_tree<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_node_count_start, d_node_count_goal,
            d_target_tree, d_node_offset_target, d_node_offset_other,
            d_current_node_count_target, d_current_node_count_other,
            d_rng_states);
        CUDA_CHECK(cudaEventRecord(step2_stop));
        
        // Step 3: Sample q_rand
        CUDA_CHECK(cudaEventRecord(step3_start));
        step3_sample_q_rand<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_q_rand_x, d_q_rand_y, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step3_stop));
        
        // Step 4: Find nearest neighbor
        CUDA_CHECK(cudaEventRecord(step4_start));
        step4_find_nearest<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_nodes, d_target_tree, d_node_offset_target,
            d_current_node_count_target, d_q_rand_x, d_q_rand_y,
            d_nearest_node_local_idx, d_nearest_node_global_idx,
            d_q_near_x, d_q_near_y, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step4_stop));
        
        // Step 5: Steer towards q_rand
        CUDA_CHECK(cudaEventRecord(step5_start));
        step5_steer<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_q_near_x, d_q_near_y, d_q_rand_x, d_q_rand_y,
            d_q_new_x, d_q_new_y, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step5_stop));
        
        // Step 6: Collision check
        CUDA_CHECK(cudaEventRecord(step6_start));
        step6_collision_check<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_q_near_x, d_q_near_y, d_q_new_x, d_q_new_y,
            d_obstacles, num_obstacles, d_collision_free, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step6_stop));
        
        // Step 7: Add node to tree
        CUDA_CHECK(cudaEventRecord(step7_start));
        step7_add_node<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_nodes, d_target_tree, d_node_count_start, d_node_count_goal,
            d_node_offset_target, d_q_new_x, d_q_new_y, d_nearest_node_global_idx,
            d_new_node_local_idx, d_new_node_global_idx, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step7_stop));
        
        // Step 8: Attempt connection
        CUDA_CHECK(cudaEventRecord(step8_start));
        step8_attempt_connection<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_thread_active, d_nodes, d_target_tree, d_node_offset_other,
            d_current_node_count_other, d_q_new_x, d_q_new_y, d_new_node_global_idx,
            d_obstacles, num_obstacles, d_connection_made, 
            d_connection_node_idx_start, d_connection_node_idx_goal, d_rng_states);
        CUDA_CHECK(cudaEventRecord(step8_stop));
        
        // Calculate elapsed time for each step
        CUDA_CHECK(cudaEventSynchronize(step1_stop));
        CUDA_CHECK(cudaEventSynchronize(step2_stop));
        CUDA_CHECK(cudaEventSynchronize(step3_stop));
        CUDA_CHECK(cudaEventSynchronize(step4_stop));
        CUDA_CHECK(cudaEventSynchronize(step5_stop));
        CUDA_CHECK(cudaEventSynchronize(step6_stop));
        CUDA_CHECK(cudaEventSynchronize(step7_stop));
        CUDA_CHECK(cudaEventSynchronize(step8_stop));
        
        CUDA_CHECK(cudaEventElapsedTime(&step1_time, step1_start, step1_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step2_time, step2_start, step2_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step3_time, step3_start, step3_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step4_time, step4_start, step4_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step5_time, step5_start, step5_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step6_time, step6_start, step6_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step7_time, step7_start, step7_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step8_time, step8_start, step8_stop));
        
        // Accumulate timing statistics
        total_timing.step1_check_connection += step1_time;
        total_timing.step2_determine_tree += step2_time;
        total_timing.step3_sample_q_rand += step3_time;
        total_timing.step4_find_nearest += step4_time;
        total_timing.step5_steer += step5_time;
        total_timing.step6_collision_check += step6_time;
        total_timing.step7_add_node += step7_time;
        total_timing.step8_attempt_connection += step8_time;
        
        // Update min/max timings
        min_timing.step1_check_connection = fminf(min_timing.step1_check_connection, step1_time);
        min_timing.step2_determine_tree = fminf(min_timing.step2_determine_tree, step2_time);
        min_timing.step3_sample_q_rand = fminf(min_timing.step3_sample_q_rand, step3_time);
        min_timing.step4_find_nearest = fminf(min_timing.step4_find_nearest, step4_time);
        min_timing.step5_steer = fminf(min_timing.step5_steer, step5_time);
        min_timing.step6_collision_check = fminf(min_timing.step6_collision_check, step6_time);
        min_timing.step7_add_node = fminf(min_timing.step7_add_node, step7_time);
        min_timing.step8_attempt_connection = fminf(min_timing.step8_attempt_connection, step8_time);
        
        max_timing.step1_check_connection = fmaxf(max_timing.step1_check_connection, step1_time);
        max_timing.step2_determine_tree = fmaxf(max_timing.step2_determine_tree, step2_time);
        max_timing.step3_sample_q_rand = fmaxf(max_timing.step3_sample_q_rand, step3_time);
        max_timing.step4_find_nearest = fmaxf(max_timing.step4_find_nearest, step4_time);
        max_timing.step5_steer = fmaxf(max_timing.step5_steer, step5_time);
        max_timing.step6_collision_check = fmaxf(max_timing.step6_collision_check, step6_time);
        max_timing.step7_add_node = fmaxf(max_timing.step7_add_node, step7_time);
        max_timing.step8_attempt_connection = fmaxf(max_timing.step8_attempt_connection, step8_time);
        
        // Periodically check if a connection has been made
        const int check_interval = 50;
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
            
            // Check if either tree is full (read counts again if not read for print)
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
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    printf("Bi-RRT computation finished in %.3f ms.\n", duration_ms.count());
    
    // Calculate average times
    if (iteration > 0) {
        avg_timing.step1_check_connection = total_timing.step1_check_connection / iteration;
        avg_timing.step2_determine_tree = total_timing.step2_determine_tree / iteration;
        avg_timing.step3_sample_q_rand = total_timing.step3_sample_q_rand / iteration;
        avg_timing.step4_find_nearest = total_timing.step4_find_nearest / iteration;
        avg_timing.step5_steer = total_timing.step5_steer / iteration;
        avg_timing.step6_collision_check = total_timing.step6_collision_check / iteration;
        avg_timing.step7_add_node = total_timing.step7_add_node / iteration;
        avg_timing.step8_attempt_connection = total_timing.step8_attempt_connection / iteration;
    }
    
    // Print timing results
    printf("\n--- Kernel Step Timing Results (over %d iterations) ---\n", iteration);
    printf("Step 1 (Check Connection):     Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step1_check_connection, min_timing.step1_check_connection, 
           max_timing.step1_check_connection, total_timing.step1_check_connection);
    printf("Step 2 (Determine Tree):       Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step2_determine_tree, min_timing.step2_determine_tree, 
           max_timing.step2_determine_tree, total_timing.step2_determine_tree);
    printf("Step 3 (Sample q_rand):        Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step3_sample_q_rand, min_timing.step3_sample_q_rand, 
           max_timing.step3_sample_q_rand, total_timing.step3_sample_q_rand);
    printf("Step 4 (Find Nearest):         Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step4_find_nearest, min_timing.step4_find_nearest, 
           max_timing.step4_find_nearest, total_timing.step4_find_nearest);
    printf("Step 5 (Steer):                Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step5_steer, min_timing.step5_steer, 
           max_timing.step5_steer, total_timing.step5_steer);
    printf("Step 6 (Collision Check):      Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step6_collision_check, min_timing.step6_collision_check, 
           max_timing.step6_collision_check, total_timing.step6_collision_check);
    printf("Step 7 (Add Node):             Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step7_add_node, min_timing.step7_add_node, 
           max_timing.step7_add_node, total_timing.step7_add_node);
    printf("Step 8 (Attempt Connection):   Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.2f ms\n",
           avg_timing.step8_attempt_connection, min_timing.step8_attempt_connection, 
           max_timing.step8_attempt_connection, total_timing.step8_attempt_connection);
           
    // Calculate total kernel time
    float total_kernel_time = 
        total_timing.step1_check_connection + 
        total_timing.step2_determine_tree +
        total_timing.step3_sample_q_rand +
        total_timing.step4_find_nearest +
        total_timing.step5_steer +
        total_timing.step6_collision_check +
        total_timing.step7_add_node +
        total_timing.step8_attempt_connection;
        
    printf("Total Kernel Time: %.2f ms\n", total_kernel_time);
    printf("---------------------------------\n");

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
        // We need nodes from index 0 to h_node_count_start
        // and nodes from MAX_NODES_PER_TREE to MAX_NODES_PER_TREE + h_node_count_goal
        printf("  Copying nodes from device...\n");
        CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, h_node_count_start * sizeof(Node), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nodes + MAX_NODES_PER_TREE,         // Destination offset on host
                             d_nodes + MAX_NODES_PER_TREE,         // Source offset on device
                             h_node_count_goal * sizeof(Node),
                             cudaMemcpyDeviceToHost));

        // Reconstruct path from start tree up to connection node
        std::vector<int> path_start_segment;
        int current_idx = h_connection_node_idx_start;
        int path_steps = 0;
        const int MAX_PATH_STEPS = h_node_count_start + h_node_count_goal + 2; // Safety break

        printf("  Tracing start tree segment...\n");
        while (current_idx != -1 && path_steps < MAX_PATH_STEPS) {
             if (current_idx < 0 || current_idx >= MAX_NODES_PER_TREE) { // Validate index range
                 fprintf(stderr, "Error: Invalid node index %d in start tree during path reconstruction.\n", current_idx);
                 path_start_segment.clear(); break;
             }
            path_start_segment.push_back(current_idx);
            current_idx = h_nodes[current_idx].parent_idx;
            path_steps++;
        }
        if (current_idx != -1 || path_steps >= MAX_PATH_STEPS) { // Check for failure
            fprintf(stderr, "Error: Start path reconstruction failed or exceeded max steps.\n");
            path_start_segment.clear();
        } else {
            std::reverse(path_start_segment.begin(), path_start_segment.end()); // Reverse to get start -> connection order
        }

        // Reconstruct path from goal tree up to connection node
        std::vector<int> path_goal_segment;
        current_idx = h_connection_node_idx_goal;
        path_steps = 0; // Reset step count

        printf("  Tracing goal tree segment...\n");
        while (current_idx != -1 && path_steps < MAX_PATH_STEPS) {
             // Validate index range for goal tree nodes (offset on host/device)
             if (current_idx < MAX_NODES_PER_TREE || current_idx >= MAX_TOTAL_NODES) {
                 fprintf(stderr, "Error: Invalid node index %d in goal tree during path reconstruction.\n", current_idx);
                 path_goal_segment.clear(); break;
             }
            path_goal_segment.push_back(current_idx);
            current_idx = h_nodes[current_idx].parent_idx; // Parent index is also global
            path_steps++;
        }
        if (current_idx != -1 || path_steps >= MAX_PATH_STEPS) { // Check for failure
            fprintf(stderr, "Error: Goal path reconstruction failed or exceeded max steps.\n");
            path_goal_segment.clear();
        }
        // NOTE: Goal segment is already in connection -> goal order, no need to reverse.

        // Combine the paths if both segments are valid
        if (!path_start_segment.empty() && !path_goal_segment.empty()) {
            path_indices = path_start_segment;
            path_indices.insert(path_indices.end(), path_goal_segment.begin(), path_goal_segment.end());
            printf("Path Found (%d steps).\n", (int)path_indices.size());
        } else {
            printf("Path reconstruction failed (one or both segments invalid).\n");
            h_connection_made = 0; // Mark as failed
        }

    } else {
        // Connection not made message
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
    // Get final counts if not already retrieved
    if(h_node_count_start <= 1 || h_node_count_goal <= 1) {
        CUDA_CHECK(cudaMemcpy(&h_node_count_start, d_node_count_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_node_count_goal, d_node_count_goal, sizeof(int), cudaMemcpyDeviceToHost));
    }
    // Copy nodes if connection wasn't made but we want to save the trees
    if (!h_connection_made) {
        printf("Copying nodes from device for saving trees...\n");
        if(h_node_count_start > 0)
           CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, h_node_count_start * sizeof(Node), cudaMemcpyDeviceToHost));
        if(h_node_count_goal > 0)
           CUDA_CHECK(cudaMemcpy(h_nodes + MAX_NODES_PER_TREE, d_nodes + MAX_NODES_PER_TREE, h_node_count_goal * sizeof(Node), cudaMemcpyDeviceToHost));
    }

    printf("Saving results to CSV files...\n");

    // Save node data
    FILE* nodes_file = fopen("birrt_nodes.csv", "w");
    if (nodes_file == NULL) {
        fprintf(stderr, "Error opening nodes output file 'birrt_nodes.csv'\n");
    } else {
        fprintf(nodes_file, "global_id,x,y,parent_global_id,tree_type,on_path\n"); // Added tree_type
        std::vector<bool> on_path(MAX_TOTAL_NODES, false); // Use global size
        if (h_connection_made && !path_indices.empty()) {
            for (int idx : path_indices) {
                if (idx >= 0 && idx < MAX_TOTAL_NODES) { // Check global index bounds
                    on_path[idx] = true;
                }
            }
        }

        // Write start tree nodes
        for (int i = 0; i < h_node_count_start; ++i) {
             int global_idx = i; // Start tree nodes are directly indexed
             fprintf(nodes_file, "%d,%.4f,%.4f,%d,%d,%d\n",
                     global_idx, h_nodes[global_idx].x, h_nodes[global_idx].y,
                     h_nodes[global_idx].parent_idx,
                     START_TREE_IDX, // Tree type 0
                     on_path[global_idx] ? 1 : 0);
        }
        // Write goal tree nodes
        for (int i = 0; i < h_node_count_goal; ++i) {
             int global_idx = MAX_NODES_PER_TREE + i; // Goal tree nodes are offset
             fprintf(nodes_file, "%d,%.4f,%.4f,%d,%d,%d\n",
                     global_idx, h_nodes[global_idx].x, h_nodes[global_idx].y,
                     h_nodes[global_idx].parent_idx,
                     GOAL_TREE_IDX, // Tree type 1
                     on_path[global_idx] ? 1 : 0);
        }
        fclose(nodes_file);
        printf("Saved %d start tree nodes and %d goal tree nodes to birrt_nodes.csv\n",
               h_node_count_start, h_node_count_goal);
    }

    // Save world data (Same format as before)
    FILE* world_file = fopen("birrt_world.csv", "w");
     if (world_file == NULL) {
        fprintf(stderr, "Error opening world output file 'birrt_world.csv'\n");
    } else {
        fprintf(world_file, "type,x,y,width,height_or_threshold\n");
        fprintf(world_file, "world,0,0,%.2f,%.2f\n", WORLD_WIDTH, WORLD_HEIGHT);
        fprintf(world_file, "start,%.2f,%.2f,0,0\n", START_X, START_Y);
        fprintf(world_file, "goal,%.2f,%.2f,0,0\n", GOAL_X, GOAL_Y); // Threshold not as relevant here
        for (size_t i = 0; i < h_obstacles.size(); i++) {
            fprintf(world_file, "obstacle,%.2f,%.2f,%.2f,%.2f\n",
                    h_obstacles[i].x_min, h_obstacles[i].y_min,
                    h_obstacles[i].x_max - h_obstacles[i].x_min,
                    h_obstacles[i].y_max - h_obstacles[i].y_min);
        }
        fclose(world_file);
        printf("Saved world data to birrt_world.csv\n");
    }

    // Save timing data
    FILE* timing_file = fopen("birrt_performance.csv", "w");
    if (timing_file == NULL) {
        fprintf(stderr, "Error opening timing output file 'birrt_performance.csv'\n");
    } else {
        fprintf(timing_file, "step,avg_time_ms,min_time_ms,max_time_ms,total_time_ms\n");
        fprintf(timing_file, "check_connection,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step1_check_connection, min_timing.step1_check_connection, 
                max_timing.step1_check_connection, total_timing.step1_check_connection);
        fprintf(timing_file, "determine_tree,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step2_determine_tree, min_timing.step2_determine_tree, 
                max_timing.step2_determine_tree, total_timing.step2_determine_tree);
        fprintf(timing_file, "sample_random,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step3_sample_q_rand, min_timing.step3_sample_q_rand, 
                max_timing.step3_sample_q_rand, total_timing.step3_sample_q_rand);
        fprintf(timing_file, "find_nearest,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step4_find_nearest, min_timing.step4_find_nearest, 
                max_timing.step4_find_nearest, total_timing.step4_find_nearest);
        fprintf(timing_file, "steer,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step5_steer, min_timing.step5_steer, 
                max_timing.step5_steer, total_timing.step5_steer);
        fprintf(timing_file, "collision_check,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step6_collision_check, min_timing.step6_collision_check, 
                max_timing.step6_collision_check, total_timing.step6_collision_check);
        fprintf(timing_file, "add_node,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step7_add_node, min_timing.step7_add_node, 
                max_timing.step7_add_node, total_timing.step7_add_node);
        fprintf(timing_file, "attempt_connection,%.6f,%.6f,%.6f,%.6f\n", 
                avg_timing.step8_attempt_connection, min_timing.step8_attempt_connection, 
                max_timing.step8_attempt_connection, total_timing.step8_attempt_connection);
        fprintf(timing_file, "total_kernel,%.6f,0,0,%.6f\n", 
                total_kernel_time/iteration, total_kernel_time);
        fclose(timing_file);
        printf("Saved performance timing data to birrt_performance.csv\n");
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
    
    // Free temporary arrays
    CUDA_CHECK(cudaFree(d_thread_active));
    CUDA_CHECK(cudaFree(d_target_tree));
    CUDA_CHECK(cudaFree(d_node_offset_target));
    CUDA_CHECK(cudaFree(d_node_offset_other));
    CUDA_CHECK(cudaFree(d_current_node_count_target));
    CUDA_CHECK(cudaFree(d_current_node_count_other));
    CUDA_CHECK(cudaFree(d_q_rand_x));
    CUDA_CHECK(cudaFree(d_q_rand_y));
    CUDA_CHECK(cudaFree(d_nearest_node_local_idx));
    CUDA_CHECK(cudaFree(d_nearest_node_global_idx));
    CUDA_CHECK(cudaFree(d_q_near_x));
    CUDA_CHECK(cudaFree(d_q_near_y));
    CUDA_CHECK(cudaFree(d_q_new_x));
    CUDA_CHECK(cudaFree(d_q_new_y));
    CUDA_CHECK(cudaFree(d_collision_free));
    CUDA_CHECK(cudaFree(d_new_node_local_idx));
    CUDA_CHECK(cudaFree(d_new_node_global_idx));
    
    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(step1_start));
    CUDA_CHECK(cudaEventDestroy(step1_stop));
    CUDA_CHECK(cudaEventDestroy(step2_start));
    CUDA_CHECK(cudaEventDestroy(step2_stop));
    CUDA_CHECK(cudaEventDestroy(step3_start));
    CUDA_CHECK(cudaEventDestroy(step3_stop));
    CUDA_CHECK(cudaEventDestroy(step4_start));
    CUDA_CHECK(cudaEventDestroy(step4_stop));
    CUDA_CHECK(cudaEventDestroy(step5_start));
    CUDA_CHECK(cudaEventDestroy(step5_stop));
    CUDA_CHECK(cudaEventDestroy(step6_start));
    CUDA_CHECK(cudaEventDestroy(step6_stop));
    CUDA_CHECK(cudaEventDestroy(step7_start));
    CUDA_CHECK(cudaEventDestroy(step7_stop));
    CUDA_CHECK(cudaEventDestroy(step8_start));
    CUDA_CHECK(cudaEventDestroy(step8_stop));
    
    printf("Done.\n");
    return 0;
}