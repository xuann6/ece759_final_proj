import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

def visualize_rrt():
    # Load data
    try:
        nodes_df = pd.read_csv('rrt_nodes.csv')
        world_df = pd.read_csv('rrt_world.csv')
    except FileNotFoundError:
        print("Error: CSV files not found. Make sure to run the CUDA code first.")
        return
    
    # Set the node ID as the index for easier lookups
    nodes_df = nodes_df.set_index('id')
    
    print(f"Loaded {len(nodes_df)} nodes")
    
    # Create figure and axis for the tree visualization
    plt.figure(figsize=(12, 10))
    
    # Plot connections between nodes (tree edges) - with error handling
    valid_connections = 0
    invalid_connections = 0
    
    for idx, node in nodes_df.iterrows():
        parent_id = node['parent']
        if parent_id >= 0:  # Skip the root node
            # Check if parent exists in the dataframe
            if parent_id in nodes_df.index:
                parent = nodes_df.loc[parent_id]
                plt.plot([node['x'], parent['x']], [node['y'], parent['y']], 
                         color='lightgray', alpha=0.3, linewidth=0.5)
                valid_connections += 1
            else:
                invalid_connections += 1
    
    print(f"Valid connections plotted: {valid_connections}")
    print(f"Invalid connections skipped: {invalid_connections}")
    
    # Plot nodes
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    non_path_nodes = nodes_df[nodes_df['on_path'] == 0]
    
    # Plot non-path nodes
    plt.scatter(non_path_nodes['x'], non_path_nodes['y'], 
                s=5, alpha=0.3, color='blue', label='RRT Tree')
    
    # Plot path nodes if any
    if not path_nodes.empty:
        plt.scatter(path_nodes['x'], path_nodes['y'], 
                    s=20, color='green', label='Path')
        
        # Connect path nodes
        path_indices = path_nodes.index.tolist()
        for i in range(len(path_indices) - 1):
            node1 = path_nodes.iloc[i]
            node2 = path_nodes.iloc[i+1]
            plt.plot([node1['x'], node2['x']], [node1['y'], node2['y']], 
                     color='green', linewidth=2.5)
    
    # Extract world info
    world_info = world_df[world_df['type'] == 'world'].iloc[0]
    world_width, world_height = world_info['x2'], world_info['y2']
    
    # Draw obstacles
    obstacles = world_df[world_df['type'] == 'obstacle']
    for _, obstacle in obstacles.iterrows():
        rect = Rectangle((obstacle['x1'], obstacle['y1']), 
                         obstacle['x2'], obstacle['y2'], 
                         fill=True, color='red', alpha=0.5)
        plt.gca().add_patch(rect)
    
    # Draw start and goal
    start = world_df[world_df['type'] == 'start'].iloc[0]
    goal = world_df[world_df['type'] == 'goal'].iloc[0]
    
    plt.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
    plt.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
    
    # Set axis limits
    plt.xlim(0, world_width)
    plt.ylim(0, world_height)
    plt.title('RRT Tree')
    plt.legend()
    plt.savefig('rrt_tree.png')
    
    # Create a heatmap of node density
    plt.figure(figsize=(12, 10))
    x = nodes_df['x'].values
    y = nodes_df['y'].values
    
    # Generate heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, world_width], [0, world_height]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Node Density')
    
    # Draw obstacles on heatmap
    for _, obstacle in obstacles.iterrows():
        rect = Rectangle((obstacle['x1'], obstacle['y1']), 
                         obstacle['x2'], obstacle['y2'], 
                         fill=True, color='red', alpha=0.3, edgecolor='black')
        plt.gca().add_patch(rect)
    
    # Mark start and goal on heatmap
    plt.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
    plt.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
    
    plt.title('Node Density Heatmap')
    plt.legend()
    plt.savefig('rrt_heatmap.png')
    
    # Calculate node distribution for analysis
    top_left = len(nodes_df[(nodes_df['x'] < world_width/2) & (nodes_df['y'] < world_height/2)])
    top_right = len(nodes_df[(nodes_df['x'] >= world_width/2) & (nodes_df['y'] < world_height/2)])
    bottom_left = len(nodes_df[(nodes_df['x'] < world_width/2) & (nodes_df['y'] >= world_height/2)])
    bottom_right = len(nodes_df[(nodes_df['x'] >= world_width/2) & (nodes_df['y'] >= world_height/2)])
    
    # Get obstacle info
    obstacle1 = obstacles.iloc[0]
    obstacle2 = obstacles.iloc[1]
    
    # Calculate the narrow passage area
    passage_x1 = obstacle1['x1'] + obstacle1['x2']
    passage_x2 = obstacle2['x1']
    passage_y1 = obstacle1['y2']
    passage_y2 = obstacle2['y1']
    
    print(f"Narrow passage: ({passage_x1}, {passage_y1}) to ({passage_x2}, {passage_y2})")
    
    # Count nodes in narrow passage
    narrow_passage = len(nodes_df[(nodes_df['x'] >= passage_x1) & (nodes_df['x'] <= passage_x2) &
                                 (nodes_df['y'] >= passage_y1) & (nodes_df['y'] <= passage_y2)])
    
    # Print analysis
    total_nodes = len(nodes_df)
    print(f"Total nodes: {total_nodes}")
    print(f"\nNode distribution:")
    print(f"Top-left (Start): {top_left} nodes ({100*top_left/total_nodes:.1f}%)")
    print(f"Top-right: {top_right} nodes ({100*top_right/total_nodes:.1f}%)")
    print(f"Bottom-left: {bottom_left} nodes ({100*bottom_left/total_nodes:.1f}%)")
    print(f"Bottom-right (Goal): {bottom_right} nodes ({100*bottom_right/total_nodes:.1f}%)")
    
    print(f"\nNarrow passage analysis:")
    print(f"Nodes in narrow passage: {narrow_passage} ({100*narrow_passage/total_nodes:.1f}%)")
    
    # Diagnose based on node distribution
    print("\nDiagnosis:")
    if narrow_passage < 0.05 * total_nodes:
        print("- ISSUE: Very few nodes exploring the narrow passage (<5%)")
        print("- The RRT algorithm is struggling to navigate through the tight space")
        print("- Consider using bidirectional RRT or increasing step size")
    
    if top_left > 0.7 * total_nodes:
        print("- ISSUE: Most nodes are stuck in the start region")
        print("- The algorithm is having difficulty expanding beyond the first obstacle")
    
    if bottom_right < 0.1 * total_nodes:
        print("- ISSUE: Very few nodes reaching the goal region")
        print("- Consider increasing goal bias parameter")
    
    # Create a bar chart for node distribution
    plt.figure(figsize=(10, 6))
    regions = ['Top-Left\n(Start)', 'Top-Right', 'Bottom-Left', 'Bottom-Right\n(Goal)', 'Narrow\nPassage']
    counts = [top_left, top_right, bottom_left, bottom_right, narrow_passage]
    percentages = [100*c/total_nodes for c in counts]
    
    plt.bar(regions, percentages)
    plt.ylabel('Percentage of Nodes')
    plt.title('Node Distribution by Region')
    plt.savefig('rrt_distribution.png')
    
    print("\nAnalysis images saved:")
    print("1. rrt_tree.png - RRT tree visualization")
    print("2. rrt_heatmap.png - Node density heatmap")
    print("3. rrt_distribution.png - Regional distribution chart")
    
    plt.show()

if __name__ == "__main__":
    visualize_rrt()