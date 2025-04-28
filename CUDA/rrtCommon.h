// rrtCommon.h
// Shared data structures that are used by multiple CUDA based RRT
// implementations in this repository.  Having a single definition avoids
// duplicate/ conflicting struct definitions when several algorithm headers
// are included in the same translation unit (e.g. main.cpp).

#ifndef RRT_COMMON_H
#define RRT_COMMON_H

#include <cstddef>

// Unified node representation.
// ------------------------------------------------------------
// Different algorithms need different auxiliary fields (e.g. cost for RRT*,
// tree identifier for bidirectional RRT …).  A superset of all required
// fields is therefore provided here so that the same structure can be shared
// by every algorithm while keeping the binary layout compatible.
// ------------------------------------------------------------
struct Node {
    double x{};      // 2-D position (world units)
    double y{};
    int    parent{-1}; // Index of the parent node in the global node array
    double time{0.0};  // Timestamp when the node was added – used only for
                       // visualisation.  (seconds)

    // Optional / algorithm specific -----------------------------------------
    double cost{0.0}; // Path cost from the start node (used by RRT* variants)
    int    tree{0};   // Which tree this node belongs to (0=start, 1=goal).

    // Constructors ----------------------------------------------------------
    Node() = default;

    // Standard RRT and generic helper.
    Node(double x_, double y_, int parent_ = -1, double time_ = 0.0)
        : x(x_), y(y_), parent(parent_), time(time_) {}

    // RRT* variants (requires cost field)
    Node(double x_, double y_, int parent_, double time_, double cost_)
        : x(x_), y(y_), parent(parent_), time(time_), cost(cost_) {}

    // Bidirectional RRT (requires tree identifier)
    Node(double x_, double y_, int parent_, double time_, int tree_)
        : x(x_), y(y_), parent(parent_), time(time_), tree(tree_) {}

    // Catch-all constructor when every optional field is provided.
    Node(double x_, double y_, int parent_, double time_, double cost_, int tree_)
        : x(x_), y(y_), parent(parent_), time(time_), cost(cost_), tree(tree_) {}
};


// Axis aligned rectangular obstacle.
struct Obstacle {
    double x{};
    double y{};            // Bottom-left corner
    double width{};
    double height{};

    Obstacle() = default;

    Obstacle(double x_, double y_, double width_, double height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

#endif  // RRT_COMMON_H
