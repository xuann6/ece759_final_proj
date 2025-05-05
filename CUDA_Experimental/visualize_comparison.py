import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read the RRT nodes and world data
def read_rrt_data(nodes_file, world_file):
    nodes_df = pd.read_csv(nodes_file)
    world_df = pd.read_csv(world_file)
    return nodes_df, world_df

# Function to plot the RRT tree and path
def plot_rrt(nodes_df, world_df, title, ax):
    # Extract world boundaries
    world_row = world_df[world_df['type'] == 'world'].iloc[0]
    world_width = world_row['width']
    world_height = world_row['height_or_threshold']
    
    # Extract start and goal positions
    start_row = world_df[world_df['type'] == 'start'].iloc[0]
    goal_row = world_df[world_df['type'] == 'goal'].iloc[0]
    start_x, start_y = start_row['x'], start_row['y']
    goal_x, goal_y = goal_row['x'], goal_row['y']
    
    # Extract obstacles
    obstacles = world_df[world_df['type'] == 'obstacle']
    
    # Set up plot
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_height)
    ax.set_title(title)
    
    # Plot obstacles
    for _, obstacle in obstacles.iterrows():
        rect = plt.Rectangle((obstacle['x'], obstacle['y']), 
                            obstacle['width'], 
                            obstacle['height_or_threshold'], 
                            facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # Plot start and goal
    ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Separate trees by tree_type
    start_tree = nodes_df[nodes_df['tree_type'] == 0]
    goal_tree = nodes_df[nodes_df['tree_type'] == 1]
    
    # Plot edges for start tree (blue)
    for _, node in start_tree.iterrows():
        if node['parent_global_id'] >= 0:  # Not the root node
            parent = nodes_df[nodes_df['global_id'] == node['parent_global_id']].iloc[0]
            ax.plot([node['x'], parent['x']], [node['y'], parent['y']], 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot edges for goal tree (green)
    for _, node in goal_tree.iterrows():
        if node['parent_global_id'] >= 0:  # Not the root node
            parent = nodes_df[nodes_df['global_id'] == node['parent_global_id']].iloc[0]
            ax.plot([node['x'], parent['x']], [node['y'], parent['y']], 'g-', alpha=0.3, linewidth=0.5)
    
    # Plot the path if found
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    if not path_nodes.empty:
        # Group path nodes by tree_type
        path_start = path_nodes[path_nodes['tree_type'] == 0]
        path_goal = path_nodes[path_nodes['tree_type'] == 1]
        
        # Sort path nodes by parent indices to get the correct order
        ordered_path = []
        
        # Sort start tree path
        if not path_start.empty:
            current_node = path_start.iloc[0]  # Start with the first node
            while True:
                ordered_path.append((current_node['x'], current_node['y']))
                next_nodes = path_start[path_start['parent_global_id'] == current_node['global_id']]
                if next_nodes.empty:
                    break
                current_node = next_nodes.iloc[0]
                
        # Sort goal tree path (reverse order)
        if not path_goal.empty:
            goal_path = []
            current_node = path_goal.iloc[0]  # Start with the first node from connection point
            while True:
                goal_path.append((current_node['x'], current_node['y']))
                next_nodes = path_goal[path_goal['parent_global_id'] == current_node['global_id']]
                if next_nodes.empty:
                    break
                current_node = next_nodes.iloc[0]
            
            # Add in reverse order (from goal to connection)
            ordered_path.extend(reversed(goal_path))
        
        # Plot the path
        if ordered_path:
            path_x, path_y = zip(*ordered_path)
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
    
    ax.legend()
    return ax

# Read the data for both original and optimized versions
try:
    original_nodes, original_world = read_rrt_data('birrt_nodes.csv', 'birrt_world.csv')
    optimized_nodes, optimized_world = read_rrt_data('birrt_nodes_optimized.csv', 'birrt_world_optimized.csv')
    
    # Create comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original RRT
    plot_rrt(original_nodes, original_world, 'Original BiRRT Implementation', axs[0])
    
    # Plot optimized RRT
    plot_rrt(optimized_nodes, optimized_world, 'Optimized BiRRT Implementation', axs[1])
    
    # Add info about performance
    performance_info = (
        f"Original: 1675.95 ms, {len(original_nodes)} nodes\n"
        f"Optimized: 22.91 ms, {len(optimized_nodes)} nodes\n"
        f"Speedup: 73x"
    )
    fig.suptitle(f"BiRRT Comparison\n{performance_info}", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('birrt_comparison.png', dpi=300)
    print("Visualization saved to 'birrt_comparison.png'")
    
    # Create a separate visualization for tree density comparison
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Use a scatter plot with small points to show node density
    start_tree_original = original_nodes[original_nodes['tree_type'] == 0]
    goal_tree_original = original_nodes[original_nodes['tree_type'] == 1]
    start_tree_optimized = optimized_nodes[optimized_nodes['tree_type'] == 0]
    goal_tree_optimized = optimized_nodes[optimized_nodes['tree_type'] == 1]
    
    # Original implementation density
    axs[0].scatter(start_tree_original['x'], start_tree_original['y'], c='blue', s=1, alpha=0.5, label='Start Tree')
    axs[0].scatter(goal_tree_original['x'], goal_tree_original['y'], c='green', s=1, alpha=0.5, label='Goal Tree')
    
    # Optimized implementation density
    axs[1].scatter(start_tree_optimized['x'], start_tree_optimized['y'], c='blue', s=1, alpha=0.5, label='Start Tree')
    axs[1].scatter(goal_tree_optimized['x'], goal_tree_optimized['y'], c='green', s=1, alpha=0.5, label='Goal Tree')
    
    # Set titles and layout
    axs[0].set_title(f"Original Implementation Node Distribution\n{len(original_nodes)} nodes")
    axs[1].set_title(f"Optimized Implementation Node Distribution\n{len(optimized_nodes)} nodes")
    
    # Plot obstacles, start, and goal
    for ax, world_df in zip(axs, [original_world, optimized_world]):
        # Extract world boundaries
        world_row = world_df[world_df['type'] == 'world'].iloc[0]
        world_width = world_row['width']
        world_height = world_row['height_or_threshold']
        
        # Extract start and goal positions
        start_row = world_df[world_df['type'] == 'start'].iloc[0]
        goal_row = world_df[world_df['type'] == 'goal'].iloc[0]
        start_x, start_y = start_row['x'], start_row['y']
        goal_x, goal_y = goal_row['x'], goal_row['y']
        
        # Extract obstacles
        obstacles = world_df[world_df['type'] == 'obstacle']
        
        # Set up plot
        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_height)
        
        # Plot obstacles
        for _, obstacle in obstacles.iterrows():
            rect = plt.Rectangle((obstacle['x'], obstacle['y']), 
                                obstacle['width'], 
                                obstacle['height_or_threshold'], 
                                facecolor='gray', alpha=0.5)
            ax.add_patch(rect)
        
        # Plot start and goal
        ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
        ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
        ax.legend()
    
    fig.suptitle("Node Distribution Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig('birrt_distribution_comparison.png', dpi=300)
    print("Node distribution visualization saved to 'birrt_distribution_comparison.png'")
    
    # Create a heatmap visualization
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create heatmaps for both implementations
    for ax, nodes_df, world_df, title in zip(
        axs, 
        [original_nodes, optimized_nodes], 
        [original_world, optimized_world], 
        ["Original Implementation", "Optimized Implementation"]
    ):
        # Get world dimensions
        world_row = world_df[world_df['type'] == 'world'].iloc[0]
        world_width = world_row['width']
        world_height = world_row['height_or_threshold']
        
        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            nodes_df['x'], nodes_df['y'], 
            bins=[50, 50], 
            range=[[0, world_width], [0, world_height]]
        )
        
        # Plot the heatmap
        im = ax.imshow(heatmap.T, origin='lower', extent=[0, world_width, 0, world_height], 
                      aspect='auto', cmap='viridis')
        
        # Extract obstacles, start, and goal points
        obstacles = world_df[world_df['type'] == 'obstacle']
        start_row = world_df[world_df['type'] == 'start'].iloc[0]
        goal_row = world_df[world_df['type'] == 'goal'].iloc[0]
        start_x, start_y = start_row['x'], start_row['y']
        goal_x, goal_y = goal_row['x'], goal_row['y']
        
        # Plot obstacles
        for _, obstacle in obstacles.iterrows():
            rect = plt.Rectangle((obstacle['x'], obstacle['y']), 
                                obstacle['width'], 
                                obstacle['height_or_threshold'], 
                                edgecolor='white', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        
        # Plot start and goal
        ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
        ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
        
        # Plot the path
        path_nodes = nodes_df[nodes_df['on_path'] == 1]
        if not path_nodes.empty:
            ax.plot(path_nodes['x'], path_nodes['y'], 'w-', linewidth=2, label='Path')
        
        # Set title and labels
        ax.set_title(f"{title} - Node Density\n{len(nodes_df)} nodes")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.set_label('Node Density')
    
    # Set overall title
    fig.suptitle("Node Density Heatmap Comparison", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('birrt_heatmap_comparison.png', dpi=300)
    print("Heatmap visualization saved to 'birrt_heatmap_comparison.png'")
    
except Exception as e:
    print(f"Error creating visualizations: {e}")