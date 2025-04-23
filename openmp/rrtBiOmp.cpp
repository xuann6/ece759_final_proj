#include <rrtBiOmp.h>
#include <rrtOmp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <omp.h>

namespace bidirectional_rrt_omp {

    // Find the closest pair of nodes between two trees - parallelized
    std::pair<int, int> findClosestNodes(const std::vector<Node>& treeA, const std::vector<Node>& treeB) {
        int closestA = -1;
        int closestB = -1;
        double minDist = std::numeric_limits<double>::max();
        
        // Using critical section to safely update the minimum distance
        #pragma omp parallel
        {
            // Local variables for each thread
            int local_closestA = -1;
            int local_closestB = -1;
            double local_minDist = std::numeric_limits<double>::max();
            
            // Parallelize the outer loop
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < treeA.size(); i++) {
                for (size_t j = 0; j < treeB.size(); j++) {
                    double dist = distance(treeA[i], treeB[j]);
                    if (dist < local_minDist) {
                        local_minDist = dist;
                        local_closestA = i;
                        local_closestB = j;
                    }
                }
            }
            
            // Update global minimum using critical section
            #pragma omp critical
            {
                if (local_minDist < minDist) {
                    minDist = local_minDist;
                    closestA = local_closestA;
                    closestB = local_closestB;
                }
            }
        }
        
        return std::make_pair(closestA, closestB);
    }
    
    // Check if path between two nodes is collision-free - parallelized
    bool isPathClear(const Node& from, const Node& to, 
                     const std::vector<std::vector<double>>& obstacles) {
        bool pathClear = true;
        
        // Each thread will process a subset of obstacles
        #pragma omp parallel for reduction(&& : pathClear)
        for (size_t i = 0; i < obstacles.size(); i++) {
            const auto& obstacle = obstacles[i];
            
            // Skip processing if path is already found to be blocked
            if (!pathClear) continue;
            
            // Assuming obstacles are defined as circles with [x, y, radius]
            if (obstacle.size() >= 3) {
                double obsX = obstacle[0];
                double obsY = obstacle[1];
                double obsRadius = obstacle[2];
                
                // Check if path from->to intersects with obstacle
                // Using line-circle intersection check
                
                // Vector from start to end of the line segment
                double dx = to.x - from.x;
                double dy = to.y - from.y;
                
                // Vector from start of line to center of obstacle
                double fx = from.x - obsX;
                double fy = from.y - obsY;
                
                // Calculate coefficients of quadratic equation
                double a = dx * dx + dy * dy;
                double b = 2 * (fx * dx + fy * dy);
                double c = fx * fx + fy * fy - obsRadius * obsRadius;
                
                // Calculate discriminant
                double discriminant = b * b - 4 * a * c;
                
                // If discriminant is negative, line doesn't intersect circle
                if (discriminant >= 0) {
                    // Calculate the two solutions
                    double t1 = (-b - sqrt(discriminant)) / (2 * a);
                    double t2 = (-b + sqrt(discriminant)) / (2 * a);
                    
                    // Check if any solution is within the line segment (0 <= t <= 1)
                    if ((0 <= t1 && t1 <= 1) || (0 <= t2 && t2 <= 1)) {
                        pathClear = false;  // Path intersects obstacle
                    }
                }
            }
        }
        
        return pathClear;  // Path is clear
    }
    
    // Try to connect two trees at their closest nodes
    bool tryConnect(std::vector<Node>& treeA, std::vector<Node>& treeB, 
                    const std::vector<std::vector<double>>& obstacles,
                    double stepSize, double connectThreshold) {
        // Find closest nodes between the trees
        auto closestPair = findClosestNodes(treeA, treeB);
        int idxA = closestPair.first;
        int idxB = closestPair.second;
        
        if (idxA == -1 || idxB == -1) return false;
        
        // Check if they're already close enough to connect
        if (distance(treeA[idxA], treeB[idxB]) <= connectThreshold) {
            // Check if there's a clear path between them
            if (isPathClear(treeA[idxA], treeB[idxB], obstacles)) {
                return true;
            }
        }
        
        // Try to extend treeA towards treeB
        Node steeredNode = steer(treeA[idxA], treeB[idxB], stepSize);
        
        // Check if the path is clear
        if (isPathClear(treeA[idxA], steeredNode, obstacles)) {
            // Add the new node to treeA
            steeredNode.parent = idxA;
            steeredNode.cost = treeA[idxA].cost + distance(treeA[idxA], steeredNode);
            steeredNode.time = treeA.size();  // Use index as timestamp
            
            #pragma omp critical
            {
                // Tree modification needs to be thread-safe
                treeA.push_back(steeredNode);
            }
            
            // Check if this made the connection
            if (distance(steeredNode, treeB[idxB]) <= connectThreshold && 
                isPathClear(steeredNode, treeB[idxB], obstacles)) {
                return true;
            }
        }
        
        return false;
    }
    
    // Extend a single tree towards a random point
    bool extendTree(std::vector<Node>& tree, const Node& randomNode, 
                    const std::vector<std::vector<double>>& obstacles,
                    double stepSize) {
        // Find nearest node in the tree
        int nearestIdx = rrt_omp::findNearestParallel(tree, randomNode);
        
        if (nearestIdx < 0) return false;  // Empty tree or error
        
        // Steer from nearest towards random with maximum step size
        Node newNode = steer(tree[nearestIdx], randomNode, stepSize);
        
        // Check if the path is clear
        if (isPathClear(tree[nearestIdx], newNode, obstacles)) {
            // Add new node to the tree
            newNode.parent = nearestIdx;
            newNode.cost = tree[nearestIdx].cost + distance(tree[nearestIdx], newNode);
            newNode.time = tree.size();  // Use index as timestamp
            
            #pragma omp critical
            {
                // Tree modification needs to be thread-safe
                tree.push_back(newNode);
            }
            
            return true;
        }
        
        return false;
    }
    
    // Check if goal has been reached (tree connection in bidirectional RRT)
    bool isGoalReached(const std::vector<Node>& startTree, const std::vector<Node>& goalTree, 
                       const std::vector<std::vector<double>>& obstacles,
                       double threshold) {
        auto closestPair = findClosestNodes(startTree, goalTree);
        int startIdx = closestPair.first;
        int goalIdx = closestPair.second;
        
        if (startIdx < 0 || goalIdx < 0) return false;
        
        double dist = distance(startTree[startIdx], goalTree[goalIdx]);
        
        // Check if distance is within threshold and path is clear
        return (dist <= threshold) && isPathClear(startTree[startIdx], goalTree[goalIdx], obstacles);
    }
    
    // Merge two trees to create a complete path
    std::vector<Node> mergeTrees(const std::vector<Node>& startTree, int startConnectIndex, 
                               const std::vector<Node>& goalTree, int goalConnectIndex) {
        std::vector<Node> mergedTree;
        
        // Preallocate to avoid resizing
        mergedTree.reserve(startTree.size() + goalTree.size());
        
        // Add all nodes from start tree
        for (const auto& node : startTree) {
            mergedTree.push_back(node);
        }
        
        int startTreeSize = startTree.size();
        
        // Add all nodes from goal tree (with adjusted parent indices)
        for (size_t i = 0; i < goalTree.size(); i++) {
            Node adjustedNode = goalTree[i];
            
            // Adjust parent index for nodes from goal tree
            if (adjustedNode.parent != -1) {
                adjustedNode.parent = startTreeSize + adjustedNode.parent;
            }
            
            mergedTree.push_back(adjustedNode);
        }
        
        // Connect the trees by setting the parent of the first goal tree node
        if (mergedTree.size() > startTreeSize && goalConnectIndex >= 0 && startConnectIndex >= 0) {
            mergedTree[goalConnectIndex].parent = startConnectIndex;
        }
        
        return mergedTree;
    }
    
    // Extract the final path from merged trees
    std::vector<Node> extractBidirectionalPath(const std::vector<Node>& mergedTree, int startIndex, int goalIndex) {
        // First extract the raw path by traversing from goal to start
        std::vector<int> pathIndices;
        int currentIdx = goalIndex;
        
        while (currentIdx != -1) {
            pathIndices.push_back(currentIdx);
            currentIdx = mergedTree[currentIdx].parent;
        }
        
        // Reverse to get path from start to goal
        std::reverse(pathIndices.begin(), pathIndices.end());
        
        // Create the final path using the indices
        std::vector<Node> path;
        path.reserve(pathIndices.size());  // Preallocate for efficiency
        
        for (int idx : pathIndices) {
            path.push_back(mergedTree[idx]);
        }
        
        return path;
    }
    
    // Main Bidirectional RRT algorithm - modified for parallelism
    std::vector<Node> buildBidirectionalRRT(
        const Node& start,
        const Node& goal,
        const std::vector<std::vector<double>>& obstacles,
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
        // Initialize trees with their respective roots
        omp_set_num_threads(numThreads);
        std::vector<Node> startTree = {start};
        std::vector<Node> goalTree = {goal};
        
        // Random number generation - unique seed for each thread
        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> disX(xMin, xMax);
        std::uniform_real_distribution<> disY(yMin, yMax);
        
        bool useStartTree = true; // Flag to alternate between trees
        bool treesConnected = false;
        int connectedIteration = -1;
        
        // Set up parallel seed
        #pragma omp parallel
        {
            // Each thread gets a unique seed based on thread number
            unsigned int thread_seed = seed + omp_get_thread_num();
            std::mt19937 thread_gen(thread_seed);
        }
        
        // Execute iterations in chunks to periodically check for connections
        for (int iter = 0; iter < maxIterations && !treesConnected; iter += 10) {
            // Use task-based parallelism for batches of iterations
            #pragma omp parallel for schedule(dynamic)
            for (int batch = 0; batch < 10 && iter + batch < maxIterations; batch++) {
                // Skip if we already found a connection
                if (treesConnected) continue;
                
                int currentIter = iter + batch;
                // Thread-local random number generators
                std::mt19937 thread_gen(seed + omp_get_thread_num() + currentIter);
                std::uniform_real_distribution<> thread_disX(xMin, xMax);
                std::uniform_real_distribution<> thread_disY(yMin, yMax);
                
                // Generate random node
                Node randomNode(thread_disX(thread_gen), thread_disY(thread_gen));
                
                // Determine which tree to extend (alternate based on iteration)
                bool localUseStartTree = ((currentIter % 2) == 0);
                
                // Need critical sections when accessing the trees
                #pragma omp critical
                {
                    std::vector<Node>& currentTree = localUseStartTree ? startTree : goalTree;
                    std::vector<Node>& otherTree = localUseStartTree ? goalTree : startTree;
                    
                    // Extend the current tree
                    if (extendTree(currentTree, randomNode, obstacles, stepSize)) {
                        // Try to connect the trees
                        if (tryConnect(currentTree, otherTree, obstacles, stepSize, connectThreshold)) {
                            treesConnected = true;
                            connectedIteration = currentIter;
                        }
                    }
                }
            }
            
            // Check direct connection after each batch
            if (!treesConnected && isGoalReached(startTree, goalTree, obstacles, connectThreshold)) {
                treesConnected = true;
                connectedIteration = iter;
            }
        }
        
        // If we found a connection
        if (treesConnected) {
            auto closestPair = findClosestNodes(startTree, goalTree);
            
            // Merge trees
            std::vector<Node> mergedTree = mergeTrees(
                startTree, closestPair.first, 
                goalTree, closestPair.second
            );

            Node connectingNode(startTree[closestPair.first].x, startTree[closestPair.first].y);
            connectingNode.parent = closestPair.second;
            connectingNode.time = goalTree.size();
            goalTree.push_back(connectingNode);
            
            // Save trees for visualization if enabled
            if (enableVisualization) {
                saveTreesToFile(startTree, goalTree, treeFilename);
            }
            
            // Extract and return the final path
            return extractBidirectionalPath(
                mergedTree, 
                0,  // Start is always the first node in the start tree
                startTree.size() + goalTree.size() - 1  // Goal is the last node in the merged tree
            );
        }
        
        // If we reached max iterations without connecting the trees
        if (enableVisualization) {
            saveTreesToFile(startTree, goalTree, treeFilename);
        }
        
        // Return partial path
        return buildPartialPath(startTree, goalTree, obstacles);
    }
    
    // Build partial path from both trees when connection fails
    std::vector<Node> buildPartialPath(
        const std::vector<Node>& startTree,
        const std::vector<Node>& goalTree,
        const std::vector<std::vector<double>>& obstacles
    ) {
        std::vector<Node> partialPath;
        
        // Find closest nodes between trees
        auto closestPair = findClosestNodes(startTree, goalTree);
        int startConnectIdx = closestPair.first;
        int goalConnectIdx = closestPair.second;
        
        if (startConnectIdx < 0 || goalConnectIdx < 0) {
            return partialPath;  // Empty path
        }
        
        // Extract path from start to closest node in start tree
        std::vector<Node> startPath = extractPath(startTree, startConnectIdx);
        
        // Extract path from closest node in goal tree to goal
        std::vector<Node> goalPath = extractPath(goalTree, goalConnectIdx);
        // Reverse goal path since extractPath works backwards
        std::reverse(goalPath.begin(), goalPath.end());
        
        // Check if the closest nodes can be connected safely
        if (isPathClear(startTree[startConnectIdx], goalTree[goalConnectIdx], obstacles)) {
            // Combine paths
            partialPath = startPath;
            partialPath.insert(partialPath.end(), goalPath.begin(), goalPath.end());
        } else {
            // Return just the path from start - better to have a partial path than none
            partialPath = startPath;
        }
        
        return partialPath;
    }
    
    // Save both trees data to a single file for visualization
    void saveTreesToFile(
        const std::vector<Node>& startTree, 
        const std::vector<Node>& goalTree,
        const std::string& treeFilename
    ) {
        std::ofstream file(treeFilename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << treeFilename << std::endl;
            return;
        }
        
        // Write header in the specified format
        file << "node_id,x,y,parent_id,time" << std::endl;
        
        // Write start tree nodes
        for (size_t i = 0; i < startTree.size(); i++) {
            const Node& node = startTree[i];
            file << i << "," << node.x << "," << node.y << ","
                 << node.parent << "," << node.time << std::endl;
        }
        
        // Write goal tree nodes with adjusted node_id
        size_t startSize = startTree.size();
        for (size_t i = 0; i < goalTree.size(); i++) {
            const Node& node = goalTree[i];
            // For goal tree nodes, map the parent_id to the global index space if it's not -1
            int parentId = (node.parent != -1) ? (node.parent + startSize) : -1;
            
            file << (i + startSize) << "," << node.x << "," << node.y << ","
                 << parentId << "," << node.time << std::endl;
            
        }
        
        file.close();
    }
}