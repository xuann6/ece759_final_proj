#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <curand_kernel.h> // Include cuRAND for GPU random numbers


// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Parameters ---
// World parameters - UW Madison Campus (Engineering Hall to Chocolate Shoppe)
#define WORLD_WIDTH 2800.2f
#define WORLD_HEIGHT 1544.7f
#define STEP_SIZE 10.0f
#define START_X 822.8f  // Engineering Hall
#define START_Y 781.8f
#define GOAL_X 2216.8f   // Chocolate Shoppe
#define GOAL_Y 1070.3f
#define GOAL_THRESHOLD 15.0f
#define GOAL_THRESHOLD_SQ (GOAL_THRESHOLD * GOAL_THRESHOLD) // Use squared distance

// RRT parameters
#define GOAL_BIAS 0.1f
#define MAX_ITERATIONS 20000 // Maximum kernel launches
#define NUM_BLOCKS 4
#define MAX_NODES 2000000    // Maximum nodes in the tree (adjust based on expected size & GPU memory)

// Obstacle parameters (Using defines directly for simplicity in kernel)
#define OBSTACLE_WIDTH (WORLD_WIDTH / 10.0f)
#define OBSTACLE1_X (WORLD_WIDTH / 3.0f - OBSTACLE_WIDTH / 2.0f)
#define OBSTACLE1_Y 0.0f // Obstacle 1 starts from the bottom
#define OBSTACLE1_HEIGHT (0.6f * WORLD_HEIGHT)
#define OBSTACLE2_X (2.0f * WORLD_WIDTH / 3.0f - OBSTACLE_WIDTH / 2.0f)
#define OBSTACLE2_HEIGHT (0.6f * WORLD_HEIGHT)
#define OBSTACLE2_Y (WORLD_HEIGHT - OBSTACLE2_HEIGHT) // Obstacle 2 starts from the top

// CUDA execution parameters
#define THREADS_PER_BLOCK 128
// Calculate grid size based on desired parallelism (e.g., try to run thousands of threads)
// #define NUM_BLOCKS ( (MAX_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK ) // Can be adjusted

// --- Data Structures ---
typedef struct {
    float x;
    float y;
    int parent_idx; // Index of the parent node in the nodes array (-1 for start node)
} Node;

// Simplified Obstacle representation (AABB: Axis-Aligned Bounding Box)
typedef struct {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} Obstacle;


// --- Device Helper Functions ---

// Calculate squared Euclidean distance (faster than sqrt)
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
// Simple check: Does the new point or the line segment cross an obstacle?
// A more robust check would involve line segment-rectangle intersection tests.
__device__ bool is_collision(float x1, float y1, float x2, float y2,
                             const Obstacle* obstacles, int num_obstacles)
{
    // Basic check: Is the endpoint inside an obstacle?
     for (int i = 0; i < num_obstacles; ++i) {
        if (x2 >= obstacles[i].x_min && x2 <= obstacles[i].x_max &&
            y2 >= obstacles[i].y_min && y2 <= obstacles[i].y_max) {
            return true; // Endpoint is inside an obstacle
        }
    }

    // Basic check: Does the line segment intersect? (Simplified: Check midpoint)
    // A full line-segment intersection test is more robust but complex.
    // This simplified check often works for RRT but isn't guaranteed.
    float mid_x = (x1 + x2) / 2.0f;
    float mid_y = (y1 + y2) / 2.0f;
     for (int i = 0; i < num_obstacles; ++i) {
         if (mid_x >= obstacles[i].x_min && mid_x <= obstacles[i].x_max &&
             mid_y >= obstacles[i].y_min && mid_y <= obstacles[i].y_max) {
             // A more thorough check would test intersection with all 4 edges of the AABB
             // For now, if midpoint is inside, consider it a collision
             return true;
         }
     }


    return false; // No collision detected (by this simple check)
}


// --- CUDA Kernels ---

// Kernel to initialize cuRAND states
__global__ void initialize_rng(curandState *states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread gets a unique seed based on its ID and the overall seed
    curand_init(seed, idx, 0, &states[idx]);
}

// Main RRT Iteration Kernel
__global__ void rrt_iteration_kernel(Node* d_nodes,
                                     int* d_node_count, // Pointer to node count on device
                                     const Obstacle* d_obstacles,
                                     int num_obstacles,
                                     curandState *rng_states,
                                     int* d_goal_found,    // Flag: 0 = not found, 1 = found
                                     int* d_goal_node_idx) // Index of the node that reached the goal
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use thread-local RNG state
    curandState local_rng_state = rng_states[tid];

    // --- 1. Check if goal already found by another thread in a previous iteration/batch ---
    //    (Read once at the beginning to potentially save work)
    if (*d_goal_found == 1) {
        // Update this thread's RNG state before returning
        rng_states[tid] = local_rng_state;
        return;
    }

    // --- 2. Get current number of nodes ---
    //    Read the *current* count. Note: This might increase while the kernel runs.
    //    This is a potential source of slight non-determinism or subtle race conditions
    //    if not handled carefully, but is common in parallel RRT.
    int current_node_count = *d_node_count;
    if (current_node_count <= 0 || current_node_count >= MAX_NODES) {
         rng_states[tid] = local_rng_state; // Save RNG state
         return; // Safety check or tree is full
    }


    // --- 3. Sample q_rand ---
    float q_rand_x, q_rand_y;
    float random_val = curand_uniform(&local_rng_state);

    if (random_val < GOAL_BIAS) {
        q_rand_x = GOAL_X;
        q_rand_y = GOAL_Y;
    } else {
        q_rand_x = curand_uniform(&local_rng_state) * WORLD_WIDTH;
        q_rand_y = curand_uniform(&local_rng_state) * WORLD_HEIGHT;
    }

    // --- 4. Find Nearest Neighbor (q_near) ---
    int nearest_node_idx = -1;
    float min_dist_sq = INFINITY;

    // Linear scan through existing nodes (up to the count read at the start)
    for (int i = 0; i < current_node_count; ++i) {
        // Read node data directly from global memory
        // Potential Optimization: Use shared memory for caching if blocks work on nearby regions
        float dist_sq = distance_sq(d_nodes[i].x, d_nodes[i].y, q_rand_x, q_rand_y);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_node_idx = i;
        }
    }

    // Should always find a nearest node if current_node_count > 0
    if (nearest_node_idx == -1) {
        rng_states[tid] = local_rng_state; // Save RNG state
        return;
    }

    Node q_near = d_nodes[nearest_node_idx]; // Get the nearest node details

    // --- 5. Steer from q_near towards q_rand to get q_new ---
    float dir_x = q_rand_x - q_near.x;
    float dir_y = q_rand_y - q_near.y;
    float mag = sqrtf(dir_x * dir_x + dir_y * dir_y);
    float q_new_x, q_new_y;

    if (mag <= STEP_SIZE) { // If q_rand is closer than step_size, q_new is q_rand
        q_new_x = q_rand_x;
        q_new_y = q_rand_y;
    } else { // Steer by STEP_SIZE
        float scale = STEP_SIZE / mag;
        q_new_x = q_near.x + dir_x * scale;
        q_new_y = q_near.y + dir_y * scale;
    }

    // Clamp q_new to world boundaries
    q_new_x = clamp(q_new_x, 0.0f, WORLD_WIDTH);
    q_new_y = clamp(q_new_y, 0.0f, WORLD_HEIGHT);


    // --- 6. Collision Check ---
    if (!is_collision(q_near.x, q_near.y, q_new_x, q_new_y, d_obstacles, num_obstacles)) {

        // --- 7. Add Node to Tree (If Collision Free) ---
        // Atomically get the index for the new node
        int new_node_idx = atomicAdd(d_node_count, 1);

        if (new_node_idx < MAX_NODES) {
            // Write the new node data
            d_nodes[new_node_idx].x = q_new_x;
            d_nodes[new_node_idx].y = q_new_y;
            d_nodes[new_node_idx].parent_idx = nearest_node_idx;

            // --- 8. Goal Check ---
            float dist_to_goal_sq = distance_sq(q_new_x, q_new_y, GOAL_X, GOAL_Y);
            if (dist_to_goal_sq <= GOAL_THRESHOLD_SQ) {
                // Atomically set the goal found flag and store the index
                // atomicExch returns the *old* value. If it was 0, this thread is the first!
                if (atomicExch(d_goal_found, 1) == 0) {
                   // Optional: Ensure only one goal index is written,
                   // could use atomicCAS if multiple threads might reach goal *exactly* simultaneously.
                   // atomicExch is often sufficient as we just need *a* goal node index.
                   atomicExch(d_goal_node_idx, new_node_idx);
                }
                // Even if goal found, let the node addition complete.
            }
        } else {
            // Tree is full, revert the count increment if possible (can be tricky)
            // A simpler approach is just to stop adding, but the count will be off.
            // For robustness, could check count *before* atomicAdd, but adds complexity.
            // Safest: Let count increment, but host checks against MAX_NODES later.
             atomicSub(d_node_count, 1); // Attempt to correct count if add failed
        }
    }

    // --- Update RNG State ---
    rng_states[tid] = local_rng_state; // Store the updated state back to global memory
}


// --- Host Code ---
int main() {
    // --- Setup Obstacles from CSV file ---
    std::vector<Obstacle> h_obstacles;
    
    // Read obstacle data from CSV file
    FILE* building_file = fopen("building_obstacles.csv", "r");
    if (building_file == NULL) {
        fprintf(stderr, "Error: Cannot open building obstacles file. Using default obstacles.\n");
        // Add a few default obstacles if file can't be opened
        h_obstacles.push_back({WORLD_WIDTH / 3, WORLD_HEIGHT / 4, WORLD_WIDTH / 3 + 100, WORLD_HEIGHT / 4 + 200});
        h_obstacles.push_back({WORLD_WIDTH * 2/3, WORLD_HEIGHT * 2/3, WORLD_WIDTH * 2/3 + 150, WORLD_HEIGHT * 2/3 + 150});
    } else {
        // Skip header line
        char line[256];
        fgets(line, sizeof(line), building_file);
        
        // Read building data
        printf("Loading building obstacles from CSV file...\n");
        char name[64];
        float x_min, y_min, x_max, y_max;
        int buildings_loaded = 0;
        
        while (fscanf(building_file, "%63[^,],%f,%f,%f,%f\n", 
                      name, &x_min, &y_min, &x_max, &y_max) == 5) {
            // Add building as obstacle
            h_obstacles.push_back({x_min, y_min, x_max, y_max});
            buildings_loaded++;
            
            // Print first 5 buildings and then summarize
            if (buildings_loaded <= 5) {
                printf("  Building: %s at (%.1f,%.1f) to (%.1f,%.1f)\n", 
                       name, x_min, y_min, x_max, y_max);
            }
        }
        
        if (buildings_loaded > 5) {
            printf("  ... and %d more buildings\n", buildings_loaded - 5);
        }
        
        fclose(building_file);
        printf("Loaded %d buildings as obstacles\n", buildings_loaded);
    }
    
    int num_obstacles = h_obstacles.size();

    printf("World: %.1fx%.1f, Start: (%.1f, %.1f), Goal: (%.1f, %.1f)\n",
           WORLD_WIDTH, WORLD_HEIGHT, START_X, START_Y, GOAL_X, GOAL_Y);
    printf("Step Size: %.2f, Goal Bias: %.2f, Goal Threshold: %.2f\n", STEP_SIZE, GOAL_BIAS, GOAL_THRESHOLD);
    printf("Obstacles: %d\n", num_obstacles);
    for(int i=0; i<num_obstacles; ++i) {
        printf("  Obs %d: (%.2f, %.2f) to (%.2f, %.2f)\n", i+1,
               h_obstacles[i].x_min, h_obstacles[i].y_min, h_obstacles[i].x_max, h_obstacles[i].y_max);
    }
    printf("Max Nodes: %d, Max Iterations: %d\n", MAX_NODES, MAX_ITERATIONS);
    printf("CUDA Grid: %d blocks, %d threads/block\n", NUM_BLOCKS, THREADS_PER_BLOCK);

    // --- Host Memory Allocation ---
    Node* h_nodes = (Node*)malloc(MAX_NODES * sizeof(Node));
    if (!h_nodes) { fprintf(stderr, "Failed to allocate host memory for nodes.\n"); return 1; }

    // --- Device Memory Allocation ---
    Node* d_nodes;
    Obstacle* d_obstacles;
    curandState* d_rng_states;
    int* d_node_count;
    int* d_goal_found;
    int* d_goal_node_idx;

    CUDA_CHECK(cudaMalloc(&d_nodes, MAX_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_obstacles, num_obstacles * sizeof(Obstacle)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState))); // One state per thread
    CUDA_CHECK(cudaMalloc(&d_node_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_goal_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_goal_node_idx, sizeof(int)));

    // --- Initialization ---
    // Initialize start node
    h_nodes[0].x = START_X;
    h_nodes[0].y = START_Y;
    h_nodes[0].parent_idx = -1; // Start node has no parent
    int h_node_count = 1;
    int h_goal_found = 0;
    int h_goal_node_idx = -1;

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes, sizeof(Node), cudaMemcpyHostToDevice)); // Only copy start node
    CUDA_CHECK(cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count, &h_node_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_goal_found, &h_goal_found, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_goal_node_idx, &h_goal_node_idx, sizeof(int), cudaMemcpyHostToDevice));

    // Initialize RNG states
    initialize_rng<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_rng_states, time(0));
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for RNG init to complete

    printf("Starting RRT...\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Main RRT Loop ---
    int iteration;
    for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        // Launch the RRT kernel
        rrt_iteration_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_nodes, d_node_count, d_obstacles, num_obstacles,
            d_rng_states, d_goal_found, d_goal_node_idx);

        // Don't synchronize here yet, let GPU work.
        // Optional: Check for kernel launch errors immediately
        // CUDA_CHECK(cudaGetLastError());

        // Periodically check if the goal has been found (reduces copy overhead)
        // Or check every iteration if faster response is needed
        if (iteration % 50 == 0 || iteration == MAX_ITERATIONS - 1) {
             CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish before checking

             // Check if goal found
             CUDA_CHECK(cudaMemcpy(&h_goal_found, d_goal_found, sizeof(int), cudaMemcpyDeviceToHost));
             if (h_goal_found == 1) {
                 CUDA_CHECK(cudaMemcpy(&h_node_count, d_node_count, sizeof(int), cudaMemcpyDeviceToHost)); // Get final count
                 printf("Goal reached after %d iterations! Total nodes: %d\n", iteration + 1, h_node_count);
                 break;
             }

             // Optional: Print progress
             CUDA_CHECK(cudaMemcpy(&h_node_count, d_node_count, sizeof(int), cudaMemcpyDeviceToHost));
             printf("Iteration %d, Nodes: %d\n", iteration + 1, h_node_count);

             // Check if tree is full
             if (h_node_count >= MAX_NODES) {
                 printf("Maximum number of nodes (%d) reached.\n", MAX_NODES);
                 break;
             }
        }
    }

     // Ensure final kernel execution is complete if loop finished early
     CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    printf("RRT computation finished in %.3f ms.\n", duration_ms.count());


    // --- Path Reconstruction (if goal found) ---
    CUDA_CHECK(cudaMemcpy(&h_goal_found, d_goal_found, sizeof(int), cudaMemcpyDeviceToHost)); // Final check

    if (h_goal_found) {
        CUDA_CHECK(cudaMemcpy(&h_goal_node_idx, d_goal_node_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_node_count, d_node_count, sizeof(int), cudaMemcpyDeviceToHost)); // Get final count

        printf("Reconstructing path from goal node index: %d\n", h_goal_node_idx);

        // Copy all nodes back from device to host for path reconstruction
        CUDA_CHECK(cudaMemcpy(h_nodes, d_nodes, h_node_count * sizeof(Node), cudaMemcpyDeviceToHost));

        std::vector<Node> path;
        int current_idx = h_goal_node_idx;
        int path_steps = 0;
        const int MAX_PATH_STEPS = MAX_NODES; // Safety break

        while (current_idx != -1 && path_steps < MAX_PATH_STEPS) {
             if(current_idx < 0 || current_idx >= h_node_count) {
                 fprintf(stderr, "Error: Invalid node index %d during path reconstruction.\n", current_idx);
                 path.clear(); // Indicate failure
                 break;
             }
            path.push_back(h_nodes[current_idx]);
            current_idx = h_nodes[current_idx].parent_idx;
            path_steps++;
        }

         if(current_idx != -1 && path_steps >= MAX_PATH_STEPS) {
             fprintf(stderr, "Error: Path reconstruction exceeded max steps (possible cycle or error).\n");
             path.clear();
         }


        if (!path.empty()) {
             // Reverse the path to get start -> goal order
             std::reverse(path.begin(), path.end());

             printf("Path Found (%d steps):\n", (int)path.size());
            //  for (const auto& node : path) {
            //      printf("  (%.2f, %.2f)\n", node.x, node.y);
            //  }
         } else {
             printf("Path reconstruction failed.\n");
         }

    } else {
        CUDA_CHECK(cudaMemcpy(&h_node_count, d_node_count, sizeof(int), cudaMemcpyDeviceToHost)); // Get final count
        if (iteration == MAX_ITERATIONS) {
             printf("Goal not reached within %d iterations. Total nodes: %d\n", MAX_ITERATIONS, h_node_count);
        } else if (h_node_count >= MAX_NODES){
             printf("Goal not reached. Tree reached maximum size (%d nodes).\n", MAX_NODES);
        } else {
            printf("Goal not reached. iterations: %d, nodes %d \n", iteration, h_node_count);
        }
    }

    // --- Save results to CSV for visualization ---
    printf("Saving results to CSV files for visualization...\n");
    
    // Save node data
    FILE* nodes_file = fopen("rrt_nodes.csv", "w");
    if (nodes_file == NULL) {
        fprintf(stderr, "Error opening nodes output file\n");
    } else {
        // Write header
        fprintf(nodes_file, "id,x,y,parent,on_path\n");
        
        // Mark nodes on the path
        std::vector<bool> on_path(h_node_count, false);
        if (h_goal_found) {
            int current_idx = h_goal_node_idx;
            while (current_idx != -1) {
                if (current_idx >= 0 && current_idx < h_node_count) {
                    on_path[current_idx] = true;
                }
                current_idx = h_nodes[current_idx].parent_idx;
            }
        }
        
        // Write all nodes
        for (int i = 0; i < h_node_count; i++) {
            fprintf(nodes_file, "%d,%.4f,%.4f,%d,%d\n", 
                    i, h_nodes[i].x, h_nodes[i].y, h_nodes[i].parent_idx, 
                    on_path[i] ? 1 : 0);
        }
        fclose(nodes_file);
        printf("Saved %d nodes to rrt_nodes.csv\n", h_node_count);
    }
    
    // Save world data (boundaries, obstacles, start/goal)
    FILE* world_file = fopen("rrt_world.csv", "w");
    if (world_file == NULL) {
        fprintf(stderr, "Error opening world output file\n");
    } else {
        // Write header
        fprintf(world_file, "type,x1,y1,x2,y2\n");
        
        // Write world boundaries
        fprintf(world_file, "world,0,0,%.2f,%.2f\n", WORLD_WIDTH, WORLD_HEIGHT);
        
        // Write start and goal
        fprintf(world_file, "start,%.2f,%.2f,0,0\n", START_X, START_Y);
        fprintf(world_file, "goal,%.2f,%.2f,%.2f,0\n", GOAL_X, GOAL_Y, GOAL_THRESHOLD);
        
        // Write obstacles
        for (size_t i = 0; i < h_obstacles.size(); i++) {
            fprintf(world_file, "obstacle,%.2f,%.2f,%.2f,%.2f\n", 
                    h_obstacles[i].x_min, h_obstacles[i].y_min, 
                    h_obstacles[i].x_max - h_obstacles[i].x_min,  // width
                    h_obstacles[i].y_max - h_obstacles[i].y_min); // height
        }
        
        fclose(world_file);
        printf("Saved world data to rrt_world.csv\n");
    }

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    free(h_nodes);
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_obstacles));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_node_count));
    CUDA_CHECK(cudaFree(d_goal_found));
    CUDA_CHECK(cudaFree(d_goal_node_idx));
    CUDA_CHECK(cudaDeviceReset()); // Optional: Reset device state

    printf("Done.\n");
    return 0;
}