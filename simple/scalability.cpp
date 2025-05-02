#include "rrt.h"
#include "rrtStar.h"
#include "rrtInformed.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

// Structure to store benchmark results
struct BenchmarkResult {
    int maxIterations;
    double time;
    int nodesGenerated;
    double pathLength;
    bool pathFound;
};

// Function to run a benchmark for any algorithm
template <typename Func>
BenchmarkResult runBenchmark(Func buildAlgorithm, int maxIterations, bool enableVisualization, const std::string& name) {
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);
    BenchmarkResult result;
    result.maxIterations = maxIterations;
    result.pathFound = false;
    
    std::cout << "Running " << name << " with " << maxIterations << " max iterations..." << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto path = buildAlgorithm(maxIterations, enableVisualization);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = endTime - startTime;
    result.time = elapsed.count();
    
    if (!path.empty()) {
        result.pathFound = true;
        result.nodesGenerated = path.size();
        
        // Calculate path length
        double pathLength = 0.0;
        for (int i = 1; i < path.size(); i++) {
            pathLength += distance(path[i-1], path[i]);
        }
        result.pathLength = pathLength;
        
        std::cout << name << " path found with " << path.size() << " nodes in "
                  << result.time << " seconds" << std::endl;
        std::cout << name << " path length: " << pathLength << std::endl;
    } else {
        result.nodesGenerated = 0;
        result.pathLength = 0.0;
        std::cout << name << " failed to find a path" << std::endl;
    }
    
    return result;
}

// Function to save results to CSV
void saveResultsToCSV(const std::vector<BenchmarkResult>& rrtResults, 
                     const std::vector<BenchmarkResult>& rrtStarResults,
                     const std::vector<BenchmarkResult>& informedRRTStarResults,
                     const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Algorithm,MaxIterations,NodesGenerated,Time,PathLength,PathFound" << std::endl;
    
    // Write RRT results
    for (const auto& result : rrtResults) {
        file << "RRT," << result.maxIterations << "," << result.nodesGenerated << ","
             << result.time << "," << result.pathLength << "," << result.pathFound << std::endl;
    }
    
    // Write RRT* results
    for (const auto& result : rrtStarResults) {
        file << "RRT*," << result.maxIterations << "," << result.nodesGenerated << ","
             << result.time << "," << result.pathLength << "," << result.pathFound << std::endl;
    }
    
    // Write Informed RRT* results
    for (const auto& result : informedRRTStarResults) {
        file << "InformedRRT*," << result.maxIterations << "," << result.nodesGenerated << ","
             << result.time << "," << result.pathLength << "," << result.pathFound << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

// Function to generate a Python plotting script with dual-axis trade-off plot
void generatePlottingScript(const std::string& dataFilename) {
    std::ofstream file("plot_scalability_results.py");
    
    if (!file.is_open()) {
        std::cerr << "Failed to create plotting script" << std::endl;
        return;
    }
    
    file << "import pandas as pd\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import seaborn as sns\n";
    file << "import numpy as np\n\n";
    
    file << "# Read the data\n";
    file << "data = pd.read_csv('" << dataFilename << "')\n\n";
    
    file << "# Filter out failed paths\n";
    file << "data = data[data['PathFound'] == 1]\n\n";
    
    file << "# Create figure for iterations vs time\n";
    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.title('RRT Algorithm Scalability: Iterations vs. Time', fontsize=16)\n\n";
    
    file << "# Set color palette\n";
    file << "colors = sns.color_palette('viridis', 3)\n\n";
    
    file << "# Group data by algorithm and plot\n";
    file << "for i, alg in enumerate(['RRT', 'RRT*', 'InformedRRT*']):\n";
    file << "    alg_data = data[data['Algorithm'] == alg]\n";
    file << "    plt.plot(alg_data['MaxIterations'], alg_data['Time'], 'o-', label=alg, color=colors[i])\n\n";
    
    file << "# Set labels and style\n";
    file << "plt.xlabel('Number of Iterations', fontsize=12)\n";
    file << "plt.ylabel('Execution Time (seconds)', fontsize=12)\n";
    file << "plt.grid(True, linestyle='--', alpha=0.7)\n";
    file << "plt.legend()\n";
    file << "plt.xscale('log')  # Use log scale for better visualization\n";
    file << "plt.tight_layout()\n";
    file << "plt.savefig('iterations_vs_time.png', dpi=300)\n";
    file << "print('Plot saved as iterations_vs_time.png')\n\n";
    
    file << "# Create figure for nodes vs time\n";
    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.title('RRT Algorithm Scalability: Nodes vs. Time', fontsize=16)\n\n";
    
    file << "# Group data by algorithm and plot\n";
    file << "for i, alg in enumerate(['RRT', 'RRT*', 'InformedRRT*']):\n";
    file << "    alg_data = data[data['Algorithm'] == alg]\n";
    file << "    plt.plot(alg_data['NodesGenerated'], alg_data['Time'], 'o-', label=alg, color=colors[i])\n\n";
    
    file << "# Set labels and style\n";
    file << "plt.xlabel('Number of Nodes Generated', fontsize=12)\n";
    file << "plt.ylabel('Execution Time (seconds)', fontsize=12)\n";
    file << "plt.grid(True, linestyle='--', alpha=0.7)\n";
    file << "plt.legend()\n";
    file << "plt.tight_layout()\n";
    file << "plt.savefig('nodes_vs_time.png', dpi=300)\n";
    file << "print('Plot saved as nodes_vs_time.png')\n\n";
    
    file << "# Create figure for path quality analysis\n";
    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.title('Path Quality vs. Computation Time', fontsize=16)\n\n";
    
    file << "for i, alg in enumerate(['RRT', 'RRT*', 'InformedRRT*']):\n";
    file << "    alg_data = data[data['Algorithm'] == alg]\n";
    file << "    plt.plot(alg_data['Time'], alg_data['PathLength'], 'o-', label=alg, color=colors[i])\n\n";
    
    file << "plt.xlabel('Execution Time (seconds)', fontsize=12)\n";
    file << "plt.ylabel('Path Length', fontsize=12)\n";
    file << "plt.grid(True, linestyle='--', alpha=0.7)\n";
    file << "plt.legend()\n";
    file << "plt.tight_layout()\n";
    file << "plt.savefig('path_quality_analysis.png', dpi=300)\n";
    file << "print('Plot saved as path_quality_analysis.png')\n\n";
    
    file << "# Create figure for iterations vs path quality\n";
    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.title('Path Quality vs. Iterations', fontsize=16)\n\n";
    
    file << "for i, alg in enumerate(['RRT', 'RRT*', 'InformedRRT*']):\n";
    file << "    alg_data = data[data['Algorithm'] == alg]\n";
    file << "    plt.plot(alg_data['MaxIterations'], alg_data['PathLength'], 'o-', label=alg, color=colors[i])\n\n";
    
    file << "plt.xlabel('Number of Iterations', fontsize=12)\n";
    file << "plt.ylabel('Path Length', fontsize=12)\n";
    file << "plt.grid(True, linestyle='--', alpha=0.7)\n";
    file << "plt.xscale('log')  # Use log scale for better visualization\n";
    file << "plt.legend()\n";
    file << "plt.tight_layout()\n";
    file << "plt.savefig('iterations_vs_quality.png', dpi=300)\n";
    file << "print('Plot saved as iterations_vs_quality.png')\n\n";
    
    // NEW: Create dual-axis plot to show time-quality trade-off
    file << "# Create a dual-axis figure for the time-quality trade-off\n";
    file << "fig, ax1 = plt.subplots(figsize=(12, 7))\n";
    file << "ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis\n";
    file << "fig.suptitle('Performance Trade-off: Execution Time vs. Path Quality', fontsize=16)\n\n";
    
    file << "# Set different line styles and markers for clarity\n";
    file << "linestyles = ['-', '--', '-.']\n";
    file << "markers = ['o', 's', '^']\n\n";
    
    file << "# Add data for each algorithm\n";
    file << "for i, alg in enumerate(['RRT', 'RRT*', 'InformedRRT*']):\n";
    file << "    alg_data = data[data['Algorithm'] == alg]\n";
    file << "    # Sort by max iterations to ensure consistent ordering\n";
    file << "    alg_data = alg_data.sort_values('MaxIterations')\n";
    file << "    \n";
    file << "    # Plot execution time on the first axis (solid lines)\n";
    file << "    l1, = ax1.plot(alg_data['MaxIterations'], alg_data['Time'], \n";
    file << "                 linestyle=linestyles[i], marker=markers[i], color=colors[i],\n";
    file << "                 label=f'{alg} - Time')\n";
    file << "    \n";
    file << "    # Plot path length on the second axis (dashed lines with same color)\n";
    file << "    l2, = ax2.plot(alg_data['MaxIterations'], alg_data['PathLength'], \n";
    file << "                 linestyle=':', marker=markers[i], color=colors[i], alpha=0.7,\n";
    file << "                 label=f'{alg} - Path Length')\n\n";
    
    file << "# Configure the axes\n";
    file << "ax1.set_xlabel('Number of Iterations', fontsize=12)\n";
    file << "ax1.set_ylabel('Execution Time (seconds)', color='black', fontsize=12)\n";
    file << "ax2.set_ylabel('Path Length', color='black', fontsize=12)\n";
    file << "ax1.tick_params(axis='y', labelcolor='black')\n";
    file << "ax2.tick_params(axis='y', labelcolor='black')\n";
    file << "ax1.set_xscale('log')  # Log scale for iterations\n";
    file << "ax1.grid(True, linestyle='--', alpha=0.7)\n\n";
    
    file << "# Create a single legend for both axes\n";
    file << "lines1, labels1 = ax1.get_legend_handles_labels()\n";
    file << "lines2, labels2 = ax2.get_legend_handles_labels()\n";
    file << "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', \n";
    file << "           bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)\n\n";
    
    file << "# Add text explaining the trade-off\n";
    file << "plt.figtext(0.5, 0.01, \n";
    file << "            'Trade-off analysis: Lower execution time and shorter path length are better.\\n'\n";
    file << "            'The ideal algorithm would have low values on both axes as iterations increase.',\n";
    file << "            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n\n";
    
    file << "plt.tight_layout(rect=[0, 0.05, 1, 0.95])\n";
    file << "plt.savefig('time_quality_tradeoff.png', dpi=300, bbox_inches='tight')\n";
    file << "print('Trade-off plot saved as time_quality_tradeoff.png')\n\n";
    
    file << "plt.show()\n";
    
    file.close();
    std::cout << "Plotting script generated: plot_scalability_results.py" << std::endl;
    std::cout << "Run with: python3 plot_scalability_results.py" << std::endl;
}

int main() {
    // Configuration
    bool enableVisualization = false;  // Disable visualization for benchmarking
    std::vector<std::vector<double>> obstacles;  // Empty obstacles for consistent comparison
    
    // Define iterations to test (exponentially increasing)
    std::vector<int> iterationsToTest = {100, 200, 500, 1000, 2000, 5000, 10000};
    
    std::cout << "Starting scalability benchmark..." << std::endl;
    std::cout << "Testing iterations: ";
    for (int iter : iterationsToTest) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;
    
    // Storage for results
    std::vector<BenchmarkResult> rrtResults;
    std::vector<BenchmarkResult> rrtStarResults;
    std::vector<BenchmarkResult> informedRRTStarResults;
    
    // Common parameters
    Node start(0.1, 0.1);
    Node goal(0.9, 0.9);
    double stepSize = 0.1;
    double goalThreshold = 0.1;
    double rewireRadius = 0.5;
    double xMin = 0.0, xMax = 1.0, yMin = 0.0, yMax = 1.0;
    
    // Run benchmarks for each iteration count
    for (int maxIter : iterationsToTest) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing with " << maxIter << " max iterations" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Benchmark RRT
        // auto rrtFunc = [&](int iterations, bool vis) {
        //     return buildRRT(start, goal, stepSize, goalThreshold, iterations, 
        //                     xMin, xMax, yMin, yMax, 
        //                     "rrt_tree_" + std::to_string(iterations) + ".csv", vis);
        // };
        // rrtResults.push_back(runBenchmark(rrtFunc, maxIter, enableVisualization, "RRT"));
        
        // Benchmark RRT*
        auto rrtStarFunc = [&](int iterations, bool vis) {
            return rrt_star::buildRRTStar(start, goal, obstacles, stepSize, goalThreshold, 
                                        iterations, rewireRadius, xMin, xMax, yMin, yMax,
                                        "rrt_star_tree_" + std::to_string(iterations) + ".csv", vis, false);
        };
        rrtStarResults.push_back(runBenchmark(rrtStarFunc, maxIter, enableVisualization, "RRT*"));
        
        // Benchmark Informed RRT*
        auto informedRRTStarFunc = [&](int iterations, bool vis) {
            return rrt_informed::buildInformedRRTStar(start, goal, obstacles, stepSize, goalThreshold,
                                                   iterations, rewireRadius, xMin, xMax, yMin, yMax,
                                                   "rrt_informed_tree_" + std::to_string(iterations) + ".csv", vis, false);
        };
        informedRRTStarResults.push_back(runBenchmark(informedRRTStarFunc, maxIter, enableVisualization, "Informed RRT*"));
    }
    
    // Save results to CSV
    std::string resultsFile = "scalability_results.csv";
    saveResultsToCSV(rrtResults, rrtStarResults, informedRRTStarResults, resultsFile);
    
    // Generate plotting script
    generatePlottingScript(resultsFile);
    
    return 0;
}