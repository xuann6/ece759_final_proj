#include "rrtBidirectional.h"
#include "TimerUtils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <unordered_set>

namespace bidirectional_rrt {

    // Timer class to measure function execution times
    // class FunctionTimer {
    // private:
    //     static std::unordered_map<std::string, double> totalTimes;
    //     static std::unordered_map<std::string, int> callCounts;
        
    //     std::string functionName;
    //     std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    // public:
    //     FunctionTimer(const std::string& name) : functionName(name) {
    //         startTime = std::chrono::high_resolution_clock::now();
    //     }
        
    //     ~FunctionTimer() {
    //         auto endTime = std::chrono::high_resolution_clock::now();
    //         std::chrono::duration<double> elapsed = endTime - startTime;
    //         totalTimes[functionName] += elapsed.count();
    //         callCounts[functionName]++;
    //     }
        
    //     static void printStatistics() {
    //         std::cout << "\n--- Function Timing Statistics ---\n";
    //         double totalTime = 0.0;
            
    //         // First, calculate the total time spent in all functions
    //         for (const auto& entry : totalTimes) {
    //             if (entry.first == "buildBidirectionalRRT") {
    //                 totalTime = entry.second;
    //                 break;
    //             }
    //         }
            
    //         if (totalTime == 0.0 && !totalTimes.empty()) {
    //             // If buildBidirectionalRRT isn't found, use the sum of all function times
    //             for (const auto& entry : totalTimes) {
    //                 totalTime += entry.second;
    //             }
    //         }
            
    //         // Print statistics for each function
    //         for (const auto& entry : totalTimes) {
    //             const std::string& funcName = entry.first;
    //             double funcTotalTime = entry.second;
    //             int count = callCounts[funcName];
                
    //             std::cout << "Function: " << funcName << "\n";
    //             std::cout << "  Total calls: " << count << "\n";
    //             std::cout << "  Total time: " << funcTotalTime << " seconds\n";
    //             std::cout << "  Average time per call: " << (funcTotalTime / count) << " seconds\n";
    //             std::cout << "  Percentage of total: " << (funcTotalTime / totalTime * 100) << "%\n\n";
    //         }
    //     }
    // };

    // // Initialize static members
    // std::unordered_map<std::string, double> FunctionTimer::totalTimes;
    // std::unordered_map<std::string, int> FunctionTimer::callCounts;

    // Find the closest pair of nodes between two trees
    std::pair<int, int> findClosestNodes(const std::vector<Node>& treeA, const std::vector<Node>& treeB) {
        GlobalFunctionTimer timer("findClosestNodes");
        int closestA = -1;
        int closestB = -1;
        double minDist = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < treeA.size(); i++) {
            for (size_t j = 0; j < treeB.size(); j++) {
                double dist = distance(treeA[i], treeB[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closestA = i;
                    closestB = j;
                }
            }
        }
        
        return std::make_pair(closestA, closestB);
    }
    
    // Check if path between two nodes is collision-free
    bool isPathClear(const Node& from, const Node& to, 
                     const std::vector<std::vector<double>>& obstacles) {
        GlobalFunctionTimer timer("isPathClear");
        // For each obstacle, check if the path intersects with it
        for (const auto& obstacle : obstacles) {
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
                        return false;  // Path intersects obstacle
                    }
                }
            }
        }
        
        return true;  // Path is clear
    }
    
    // Try to connect two trees at their closest nodes
    bool tryConnect(std::vector<Node>& treeA, std::vector<Node>& treeB, 
                    const std::vector<std::vector<double>>& obstacles,
                    double stepSize, double connectThreshold) {
        GlobalFunctionTimer timer("tryConnect");
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
            treeA.push_back(steeredNode);
            
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
        GlobalFunctionTimer timer("extendTree");
        // Find nearest node in the tree
        int nearestIdx = findNearest(tree, randomNode);
        
        if (nearestIdx < 0) return false;  // Empty tree or error
        
        // Steer from nearest towards random with maximum step size
        Node newNode = steer(tree[nearestIdx], randomNode, stepSize);
        
        // Check if the path is clear
        if (isPathClear(tree[nearestIdx], newNode, obstacles)) {
            // Add new node to the tree
            newNode.parent = nearestIdx;
            newNode.cost = tree[nearestIdx].cost + distance(tree[nearestIdx], newNode);
            newNode.time = tree.size();  // Use index as timestamp
            tree.push_back(newNode);
            
            return true;
        }
        
        return false;
    }
    
    // Check if goal has been reached (tree connection in bidirectional RRT)
    bool isGoalReached(const std::vector<Node>& startTree, const std::vector<Node>& goalTree, 
                       const std::vector<std::vector<double>>& obstacles,
                       double threshold) {
        GlobalFunctionTimer timer("isGoalReached");
        auto closestPair = findClosestNodes(startTree, goalTree);
        int startIdx = closestPair.first;
        int goalIdx = closestPair.second;
        
        if (startIdx < 0 || goalIdx < 0) return false;
        
        double dist = distance(startTree[startIdx], goalTree[goalIdx]);
        
        // Check if distance is within threshold and path is clear
        return (dist <= threshold) && isPathClear(startTree[startIdx], goalTree[goalIdx], obstacles);
    }
    

std::vector<Node> mergeTrees(const std::vector<Node>& startTree, int startConnectIndex, 
                           const std::vector<Node>& goalTree, int goalConnectIndex) {
    GlobalFunctionTimer timer("mergeTrees");
    std::vector<Node> mergedTree;

    std::vector<int> startPath;
    int currentIdx = startConnectIndex; // goalTree中的第0个节点是goal
    std::vector<bool> startVisited(startTree.size(), false);  // 使用vector<bool>替代set
    while (currentIdx != -1 && !startVisited[currentIdx]) {
        startVisited[currentIdx] = true;
        startPath.push_back(currentIdx);
        
        if (currentIdx == 0) break; // 到达连接点，停止
        
        currentIdx = startTree[currentIdx].parent;
    }

    // 3. 将goalPath反向添加到mergedTree（除了连接点）
    int startSize = mergedTree.size();
    
    // 从连接点（不包括）到goal反向添加
    for (int i = startPath.size() - 1; i >= 0 ; i--) {
        Node node = startTree[startPath[i]];
        
        // 第一个添加的节点（连接点后面的节点）连接到startTree的连接点
        if (i == startPath.size() - 1) {
            node.parent = -1;
        } else {
            // 其他节点指向前一个添加的节点
            node.parent = mergedTree.size() - 1;
        }
        
        mergedTree.push_back(node);
    }
    
    // 2. 从goalTree中提取路径 - 从goal到连接点
    std::vector<int> goalPath;
    currentIdx = goalConnectIndex; // goalTree中的第0个节点是goal
    
    // 找到从goal到连接点的路径
    std::vector<bool> visited(goalTree.size(), false);  // 使用vector<bool>替代set
    while (currentIdx != -1 && !visited[currentIdx]) {
        visited[currentIdx] = true;
        goalPath.push_back(currentIdx);
        
        if (currentIdx == 0) break; // 到达连接点，停止
        
        currentIdx = goalTree[currentIdx].parent;
    }
    
    // 3. 将goalPath反向添加到mergedTree（除了连接点）
    startSize = mergedTree.size();
    
    // 从连接点（不包括）到goal反向添加
    for (int i = 0; i <= goalPath.size() - 1; i++) {
        Node node = goalTree[goalPath[i]];
        
        // 第一个添加的节点（连接点后面的节点）连接到startTree的连接点
        if (i == 0) {
            node.parent = startConnectIndex;
        } else {
            // 其他节点指向前一个添加的节点
            node.parent = mergedTree.size() - 1;
        }
        
        mergedTree.push_back(node);
    }
    
    return mergedTree;
}

    // 打印合并树的函数
void printMergedTree(const std::vector<Node>& mergedTree) {
    std::cout << "\n--- Merged Tree Structure ---\n";
    std::cout << "Index\tX\tY\tParent\tCost\tTime\n";
    std::cout << "------------------------------------------\n";
    
    for (size_t i = 0; i < mergedTree.size(); i++) {
        const Node& node = mergedTree[i];
        std::cout << i << "\t" 
                  << node.x << "\t" 
                  << node.y << "\t" 
                  << node.parent << "\t"
                  << node.cost << "\t"
                  << node.time << std::endl;
    }

}
    
    // Extract the final path from merged trees
    std::vector<Node> extractBidirectionalPath(const std::vector<Node>& mergedTree, int startIndex, int goalIndex) {
        GlobalFunctionTimer timer("extractBidirectionalPath");
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
        for (int idx : pathIndices) {
            path.push_back(mergedTree[idx]);
        }
        
        return path;
    }
    
    // Main Bidirectional RRT algorithm
    // Main Bidirectional RRT algorithm
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
        bool enableVisualization
    ) {
        GlobalFunctionTimer::reset();
        GlobalFunctionTimer timer("buildBidirectionalRRT");
        // Initialize trees with their respective roots
        std::vector<Node> startTree = {start};
        std::vector<Node> goalTree = {goal};
        
        // Random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> disX(xMin, xMax);
        std::uniform_real_distribution<> disY(yMin, yMax);
        
        bool useStartTree = true; // Flag to alternate between trees
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Generate random node
            Node randomNode(disX(gen), disY(gen));
            
            // Determine which tree to extend
            std::vector<Node>& currentTree = useStartTree ? startTree : goalTree;
            std::vector<Node>& otherTree = useStartTree ? goalTree : startTree;
            
            // Extend the current tree
            if (extendTree(currentTree, randomNode, obstacles, stepSize)) {
                // Try to connect the trees
                if (tryConnect(currentTree, otherTree, obstacles, stepSize, connectThreshold)) {
                    
                    auto closestPair = findClosestNodes(startTree, goalTree);
                    int startConnectIdx = closestPair.first;
                    int goalConnectIdx = closestPair.second;

                    std::vector<Node> mergedTree = mergeTrees(
                        startTree, startConnectIdx, 
                        goalTree, goalConnectIdx
                    );

                    printMergedTree(mergedTree);

                    Node connectingNode(startTree[closestPair.first].x, startTree[closestPair.first].y);
                    connectingNode.parent = closestPair.second;
                    connectingNode.time = goalTree.size();
                    goalTree.push_back(connectingNode);
                    
                    // Save trees for visualization if enabled
                    if (enableVisualization) {
                        saveTreesToFile(startTree, goalTree, treeFilename);
                    }
                    
                    // Print timing statistics
                    GlobalFunctionTimer::printStatistics();

                    return mergedTree;
                }
            }
            
            // Alternate between trees
            useStartTree = !useStartTree;
            
            // Check direct connection periodically (every 10 iterations)
            if (iter % 10 == 0) {
                if (isGoalReached(startTree, goalTree, obstacles, connectThreshold)) {
                    
                    auto closestPair = findClosestNodes(startTree, goalTree);
                    int startConnectIdx = closestPair.first;
                    int goalConnectIdx = closestPair.second;

                    std::vector<Node> mergedTree = mergeTrees(
                        startTree, startConnectIdx, 
                        goalTree, goalConnectIdx
                    );

                    printMergedTree(mergedTree);

                    Node connectingNode(startTree[closestPair.first].x, startTree[closestPair.first].y);
                    connectingNode.parent = closestPair.second;
                    connectingNode.time = goalTree.size();
                    goalTree.push_back(connectingNode);
                    
                    // Save trees for visualization if enabled
                    if (enableVisualization) {
                        saveTreesToFile(startTree, goalTree, treeFilename);
                    }
                    
                    // Print timing statistics
                    GlobalFunctionTimer::printStatistics();
                    
                    return mergedTree;
                }
            }
        }
        
        // If we reached max iterations without connecting the trees
        if (enableVisualization) {
            saveTreesToFile(startTree, goalTree, treeFilename);
        }
        
        // Print timing statistics
        GlobalFunctionTimer::printStatistics();
        
        // Return partial path
        return buildPartialPath(startTree, goalTree, obstacles);
    }
    
    // Build partial path from both trees when connection fails
    std::vector<Node> buildPartialPath(
        const std::vector<Node>& startTree,
        const std::vector<Node>& goalTree,
        const std::vector<std::vector<double>>& obstacles
    ) {
        GlobalFunctionTimer timer("buildPartialPath");
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
        GlobalFunctionTimer timer("saveTreesToFile");
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