import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os

def read_tree_data(file_path):
    if not os.path.exists(file_path):
        print(f"Tree file {file_path} not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        # Convert scientific notation time values to float
        df['time'] = df['time'].astype(float)
        return df
    except Exception as e:
        print(f"Error reading tree file: {e}")
        return None

def read_obstacles_data(file_path):
    if not os.path.exists(file_path):
        print(f"Obstacles file {file_path} not found.")
        return None
    try:
        # Read the first two lines to get world dimensions
        world_dims = pd.read_csv(file_path, nrows=1)
        
        # Read the rest of the file starting from line 3 (after the header for obstacles)
        obstacles = pd.read_csv(file_path, skiprows=2)
        
        # Combine world dimensions with obstacles
        # This ensures we keep the world_width and world_height in the dataframe
        result = pd.concat([world_dims, obstacles], ignore_index=True)
        
        return result
    except Exception as e:
        print(f"Error reading obstacles file: {e}")
        return None

def visualize_tree(tree_df, obstacles_df=None, title="RRT Visualization", animate=True, save_animation=False):
    if tree_df is None:
        return
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis limits and labels based on world dimensions from obstacles file
    world_width = 1
    world_height = 1
    if obstacles_df is not None and 'world_width' in obstacles_df.columns and 'world_height' in obstacles_df.columns:
        # Get world dimensions from the first row
        world_width = obstacles_df.loc[0, 'world_width']
        world_height = obstacles_df.loc[0, 'world_height']
    
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True)
    
    # Draw obstacles if available
    if obstacles_df is not None:
        for idx, obstacle in obstacles_df.iterrows():
            # Skip rows containing world dimensions
            if 'width' in obstacle and 'height' in obstacle and 'x' in obstacle and 'y' in obstacle:
                rect = patches.Rectangle(
                    (obstacle['x'], obstacle['y']), 
                    obstacle['width'], 
                    obstacle['height'],
                    linewidth=1, 
                    edgecolor='r', 
                    facecolor='r', 
                    alpha=0.3
                )
                ax.add_patch(rect)
    
    # Extract start and goal nodes
    start_node = tree_df.iloc[0]
    start_x, start_y = start_node['x'], start_node['y']
    
    # The goal node is the last node with parent_id not -1
    goal_node = tree_df.iloc[-1]
    goal_x, goal_y = goal_node['x'], goal_node['y']
    
    # Plot start and goal positions
    ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Add legend
    ax.legend()
    
    if animate:
        # Get the unique time values for animation
        time_values = sorted(tree_df['time'].unique())
        # Convert time values to milliseconds for display
        time_ms = [t * 1000 for t in time_values]
        print(f"Number of unique time steps: {len(time_values)}")
        print(f"Time range: {time_ms[0]:.2f} ms to {time_ms[-1]:.2f} ms")
        
        # Create a collection for nodes
        nodes_scatter = ax.scatter([], [], c='b', s=20, alpha=0.5)
        
        # Create a text object for the time display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12)
        
        # Animation update function
        def update(frame):
            # Get the current time value
            if frame < len(time_values):
                current_time = time_values[frame]
                current_time_ms = time_ms[frame]
            else:
                current_time = time_values[-1]
                current_time_ms = time_ms[-1]
            
            # Get nodes up to the current time
            visible_df = tree_df[tree_df['time'] <= current_time]
            
            # Clear previous frame's lines
            for line in ax.lines[2:]:  # Keep start and goal points
                line.remove()
            
            # Update nodes
            nodes_scatter.set_offsets(visible_df[['x', 'y']].values)
            
            # Draw edges
            lines = []
            for _, node in visible_df.iterrows():
                if node['parent_id'] >= 0:
                    parent_idx = int(node['parent_id'])
                    if parent_idx < len(tree_df):
                        parent = tree_df.iloc[parent_idx]
                        line = ax.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)[0]
                        lines.append(line)
            
            # Draw the path if the goal has been reached
            path_line = None
            if goal_node['node_id'] in visible_df['node_id'].values:
                path_nodes = []
                current_id = goal_node['node_id']
                
                # Trace back from goal to start
                while current_id != -1:
                    current_node = tree_df[tree_df['node_id'] == current_id].iloc[0]
                    path_nodes.append((current_node['x'], current_node['y']))
                    current_id = current_node['parent_id']
                
                # Extract x and y coordinates
                path_x, path_y = zip(*path_nodes)
                path_line = ax.plot(path_x, path_y, 'g-', linewidth=2)[0]
                lines.append(path_line)
            
            # Update the title with current time
            ax.set_title(f"{title} - Time: {current_time_ms:.2f} ms")
            time_text.set_text(f"Time: {current_time_ms:.2f} ms")
            
            return [nodes_scatter, time_text] + lines
        
        # Create animation with appropriate number of frames
        num_frames = min(100, len(time_values))
        frame_indices = np.linspace(0, len(time_values) - 1, num_frames).astype(int)
        
        anim = animation.FuncAnimation(
            fig, update, frames=frame_indices,
            interval=50, blit=False, repeat=False
        )
        
        # Always display the animation first
        print("Displaying animation...")
        plt.show()
        
        if save_animation:
            # Save as MP4
            writer = animation.FFMpegWriter(fps=20, bitrate=2000)
            try:
                print("Now saving animation to 'rrt_animation.mp4'...")
                # Create a new figure for saving since the original window was closed
                new_fig, new_ax = plt.subplots(figsize=(10, 10))
                
                # Set up the new figure exactly like the original
                new_ax.set_xlim(0, world_width)
                new_ax.set_ylim(0, world_height)
                new_ax.set_xlabel('X')
                new_ax.set_ylabel('Y')
                new_ax.set_title(title)
                new_ax.grid(True)
                
                # Draw obstacles if available
                if obstacles_df is not None:
                    for idx, obstacle in obstacles_df.iterrows():
                        # Skip rows containing world dimensions
                        if 'width' in obstacle and 'height' in obstacle and 'x' in obstacle and 'y' in obstacle:
                            rect = patches.Rectangle(
                                (obstacle['x'], obstacle['y']), 
                                obstacle['width'], 
                                obstacle['height'],
                                linewidth=1, 
                                edgecolor='r', 
                                facecolor='r', 
                                alpha=0.3
                            )
                            new_ax.add_patch(rect)
                
                # Plot start and goal positions
                new_ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
                new_ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
                new_ax.legend()
                
                # Initialize empty collections for the saving animation
                new_nodes_scatter = new_ax.scatter([], [], c='b', s=20, alpha=0.5)
                new_time_text = new_ax.text(0.02, 0.98, '', transform=new_ax.transAxes, fontsize=12)
                
                # Define update function for the new animation
                def new_update(frame):
                    # Get the current time value
                    if frame < len(time_values):
                        current_time = time_values[frame]
                        current_time_ms = time_ms[frame]
                    else:
                        current_time = time_values[-1]
                        current_time_ms = time_ms[-1]
                    
                    # Get nodes up to the current time
                    visible_df = tree_df[tree_df['time'] <= current_time]
                    
                    # Clear previous frame's lines
                    for line in new_ax.lines[2:]:  # Keep start and goal points
                        line.remove()
                    
                    # Update nodes
                    new_nodes_scatter.set_offsets(visible_df[['x', 'y']].values)
                    
                    # Draw edges
                    lines = []
                    for _, node in visible_df.iterrows():
                        if node['parent_id'] >= 0:
                            parent_idx = int(node['parent_id'])
                            if parent_idx < len(tree_df):
                                parent = tree_df.iloc[parent_idx]
                                line = new_ax.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)[0]
                                lines.append(line)
                    
                    # Draw the path if the goal has been reached
                    path_line = None
                    if goal_node['node_id'] in visible_df['node_id'].values:
                        path_nodes = []
                        current_id = goal_node['node_id']
                        
                        while current_id != -1:
                            current_node = tree_df[tree_df['node_id'] == current_id].iloc[0]
                            path_nodes.append((current_node['x'], current_node['y']))
                            current_id = current_node['parent_id']
                        
                        path_x, path_y = zip(*path_nodes)
                        path_line = new_ax.plot(path_x, path_y, 'g-', linewidth=2)[0]
                        lines.append(path_line)
                    
                    # Update the title with current time
                    new_ax.set_title(f"{title} - Time: {current_time_ms:.2f} ms")
                    new_time_text.set_text(f"Time: {current_time_ms:.2f} ms")
                    
                    return [new_nodes_scatter, new_time_text] + lines
                
                new_anim = animation.FuncAnimation(
                    new_fig, new_update, frames=frame_indices,
                    interval=50, blit=False, repeat=False
                )
                new_anim.save('rrt_animation.mp4', writer=writer)
                print("Animation saved successfully!")
                plt.close(new_fig)
            except Exception as e:
                print(f"Error saving animation: {e}")
    else:
        # Draw the final tree
        for _, node in tree_df.iterrows():
            if node['parent_id'] >= 0:
                parent_idx = int(node['parent_id'])
                if parent_idx < len(tree_df):
                    parent = tree_df.iloc[parent_idx]
                    ax.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)
        
        # Draw nodes
        ax.scatter(tree_df['x'], tree_df['y'], c='b', s=20, alpha=0.5)
        
        # Extract and draw the final path
        path_nodes = []
        current_id = goal_node['node_id']
        
        # Check if goal was reached
        if current_id in tree_df['node_id'].values:
            # Trace back from goal to start
            while current_id != -1:
                current_node = tree_df[tree_df['node_id'] == current_id].iloc[0]
                path_nodes.append((current_node['x'], current_node['y']))
                current_id = current_node['parent_id']
            
            # Extract x and y coordinates
            path_x, path_y = zip(*path_nodes)
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
            ax.legend()
        
        plt.show()

def main():
    # File paths for the RRT data
    tree_file = "rrt_omp_obstacles_tree.csv"
    obstacles_file = "rrt_omp_obstacles.csv"
    
    # Read the data
    tree_df = read_tree_data(tree_file)
    obstacles_df = read_obstacles_data(obstacles_file)
    
    # Set to True to save the animation, False to only display it
    save_animation = False
    
    if tree_df is not None:
        print(f"Tree data loaded with {len(tree_df)} nodes.")
        
        if obstacles_df is not None:
            print(f"Obstacles data loaded with {len(obstacles_df) - 1} obstacles.")
        
        # Visualize the tree with obstacles
        visualize_tree(tree_df, obstacles_df, title="RRT with Obstacles", animate=True, save_animation=save_animation)

if __name__ == "__main__":
    main() 