import pandas as pd
import matplotlib.pyplot as plt

def verify_solution(nodes_path, world_path, title):
    # Read data
    nodes_df = pd.read_csv(nodes_path)
    world_df = pd.read_csv(world_path)
    
    # Extract world, start, goal, and obstacle info
    world_info = world_df[world_df['type'] == 'world'].iloc[0]
    start_info = world_df[world_df['type'] == 'start'].iloc[0]
    goal_info = world_df[world_df['type'] == 'goal'].iloc[0]
    obstacles = world_df[world_df['type'] == 'obstacle']
    
    # Extract path nodes
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    
    # Set up plot
    plt.figure(figsize=(10, 10))
    plt.xlim(0, world_info['width'])
    plt.ylim(0, world_info['height'])
    plt.title(f"{title}\nNodes: {len(nodes_df)}, Path Length: {len(path_nodes)}")
    
    # Plot obstacles
    for _, obs in obstacles.iterrows():
        rect = plt.Rectangle((obs['x'], obs['y']), 
                            obs['width'], 
                            obs['height_or_threshold'],
                            facecolor='gray', alpha=0.5)
        plt.gca().add_patch(rect)
    
    # Plot start and goal
    plt.plot(start_info['x'], start_info['y'], 'go', markersize=10, label='Start')
    plt.plot(goal_info['x'], goal_info['y'], 'ro', markersize=10, label='Goal')
    
    # Plot tree edges (with reduced opacity and random sampling for clarity)
    # Start tree (blue)
    start_tree = nodes_df[nodes_df['tree_type'] == 0].sample(min(1000, len(nodes_df[nodes_df['tree_type'] == 0])))
    for _, node in start_tree.iterrows():
        if node['parent_global_id'] != -1:  # Not the root
            parent = nodes_df[nodes_df['global_id'] == node['parent_global_id']]
            if not parent.empty:
                plt.plot([node['x'], parent.iloc[0]['x']], 
                         [node['y'], parent.iloc[0]['y']], 
                         'b-', alpha=0.1, linewidth=0.5)
    
    # Goal tree (green)
    goal_tree = nodes_df[nodes_df['tree_type'] == 1].sample(min(1000, len(nodes_df[nodes_df['tree_type'] == 1])))
    for _, node in goal_tree.iterrows():
        if node['parent_global_id'] != -1:  # Not the root
            parent = nodes_df[nodes_df['global_id'] == node['parent_global_id']]
            if not parent.empty:
                plt.plot([node['x'], parent.iloc[0]['x']], 
                         [node['y'], parent.iloc[0]['y']], 
                         'g-', alpha=0.1, linewidth=0.5)
    
    # Plot path
    if not path_nodes.empty:
        for _, node in path_nodes.iterrows():
            if node['parent_global_id'] != -1:  # Not the root
                parent = nodes_df[nodes_df['global_id'] == node['parent_global_id']]
                if not parent.empty:
                    plt.plot([node['x'], parent.iloc[0]['x']], 
                             [node['y'], parent.iloc[0]['y']], 
                             'r-', linewidth=2)
    
    plt.legend(['Start', 'Goal', 'Start Tree', 'Goal Tree', 'Path'])
    
    # Save the figure
    output_path = f"{title.lower().replace(' ', '_')}_verification.png"
    plt.savefig(output_path, dpi=300)
    print(f"Verification plot saved to {output_path}")
    
    # Verify path integrity
    path_integrity_check(path_nodes, nodes_df, start_info, goal_info)

def path_integrity_check(path_nodes, nodes_df, start_info, goal_info):
    """Check if the path is continuous and connects start to goal"""
    if path_nodes.empty:
        print("WARNING: No path nodes found!")
        return
    
    # Check if path goes from start to goal
    start_tree_path = path_nodes[path_nodes['tree_type'] == 0]
    goal_tree_path = path_nodes[path_nodes['tree_type'] == 1]
    
    if start_tree_path.empty or goal_tree_path.empty:
        print("WARNING: Path doesn't include both trees!")
        return
    
    # Find start node (node with parent_idx = -1 in start tree)
    start_node = None
    for _, node in start_tree_path.iterrows():
        parent_idx = node['parent_global_id']
        if parent_idx == -1:
            start_node = node
            break
    
    # Find goal node (node with parent_idx = -1 in goal tree)
    goal_node = None
    for _, node in goal_tree_path.iterrows():
        parent_idx = node['parent_global_id']
        if parent_idx == -1:
            goal_node = node
            break
    
    # Check start and goal nodes
    if start_node is None:
        print("WARNING: Start node not found in path!")
    else:
        start_dist = ((start_node['x'] - start_info['x'])**2 + 
                      (start_node['y'] - start_info['y'])**2)**0.5
        print(f"Start node distance from start position: {start_dist:.4f}")
    
    if goal_node is None:
        print("WARNING: Goal node not found in path!")
    else:
        goal_dist = ((goal_node['x'] - goal_info['x'])**2 + 
                     (goal_node['y'] - goal_info['y'])**2)**0.5
        print(f"Goal node distance from goal position: {goal_dist:.4f}")
    
    # Check path continuity by traversing from one end
    print("\nChecking path continuity...")
    
    # Check connection between trees
    # Find the connecting nodes (where the two trees meet)
    # These are the leaf nodes of each tree in the path
    start_tree_leaf = None
    for _, node in start_tree_path.iterrows():
        child_exists = False
        for _, other_node in start_tree_path.iterrows():
            if other_node['parent_global_id'] == node['global_id']:
                child_exists = True
                break
        if not child_exists:
            start_tree_leaf = node
            break
    
    goal_tree_leaf = None
    for _, node in goal_tree_path.iterrows():
        child_exists = False
        for _, other_node in goal_tree_path.iterrows():
            if other_node['parent_global_id'] == node['global_id']:
                child_exists = True
                break
        if not child_exists:
            goal_tree_leaf = node
            break
    
    if start_tree_leaf is not None and goal_tree_leaf is not None:
        connection_dist = ((start_tree_leaf['x'] - goal_tree_leaf['x'])**2 + 
                           (start_tree_leaf['y'] - goal_tree_leaf['y'])**2)**0.5
        print(f"Distance between connecting nodes: {connection_dist:.4f}")
        if connection_dist > 0.2:  # Using 2x step size as threshold
            print("WARNING: Trees don't appear to be properly connected!")
    else:
        print("WARNING: Could not identify connecting nodes between trees!")

    # Print overall verification result
    print("\nPath verification complete")

# Verify optimized solution
try:
    verify_solution('birrt_nodes_optimized.csv', 'birrt_world_optimized.csv', 'Optimized BiRRT')
    print("\nSolution verification complete!")
except Exception as e:
    print(f"Error verifying solution: {e}")