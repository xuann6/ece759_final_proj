import requests
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
import pandas as pd

def fetch_buildings(south, west, north, east):
    """
    Fetch building data from OpenStreetMap using the Overpass API
    
    Args:
        south, west, north, east: Bounding box coordinates (latitude/longitude)
    
    Returns:
        Data from Overpass API
    """
    # Overpass API query to get buildings
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    
    print(f"Fetching buildings within bounds: {south},{west} to {north},{east}")
    response = requests.post(overpass_url, data=overpass_query)
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return None
    
    data = response.json()
    print(f"Received data with {len(data.get('elements', []))} elements")
    return data

def extract_buildings(data):
    """
    Extract building polygons from Overpass API data
    
    Args:
        data: JSON data from Overpass API
    
    Returns:
        List of buildings with name and coordinates
    """
    if not data or 'elements' not in data:
        return []
    
    buildings = []
    nodes = {}
    
    # First, collect all nodes
    for element in data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = (element['lat'], element['lon'])
    
    # Then process ways (buildings)
    for element in data['elements']:
        if element['type'] == 'way' and 'tags' in element and 'building' in element['tags']:
            name = element['tags'].get('name', f"Building-{element['id']}")
            
            # Get coordinates for this building
            coords = []
            for node_id in element['nodes']:
                if node_id in nodes:
                    coords.append(nodes[node_id])
            
            if coords:
                # Calculate bounding box
                lats = [c[0] for c in coords]
                lons = [c[1] for c in coords]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                buildings.append({
                    'name': name,
                    'polygon': coords,
                    'min_lat': min_lat,
                    'min_lon': min_lon,
                    'max_lat': max_lat,
                    'max_lon': max_lon
                })
    
    print(f"Extracted {len(buildings)} buildings")
    return buildings

def convert_to_local_coordinates(buildings, ref_lat, ref_lon, x_offset=300, y_offset=300):
    """
    Convert geographic coordinates to local coordinates
    
    Args:
        buildings: List of building dict with lat/lon coordinates
        ref_lat, ref_lon: Reference point (origin in local coordinates)
        x_offset, y_offset: Offsets to ensure positive coordinates
    
    Returns:
        List of buildings with local coordinates
    """
    # Earth's radius in meters
    R = 6371000
    
    # Conversion factors
    lat_to_meter = 111320  # 1 degree latitude = ~111320 meters
    lon_to_meter = 111320 * np.cos(np.radians(ref_lat))  # 1 degree longitude = ~111320*cos(lat) meters
    
    local_buildings = []
    for building in buildings:
        # Convert polygon coordinates
        local_polygon = []
        for lat, lon in building['polygon']:
            x = (lon - ref_lon) * lon_to_meter + x_offset
            y = (lat - ref_lat) * lat_to_meter + y_offset
            local_polygon.append((x, y))
        
        # Convert bounding box
        x_min = (building['min_lon'] - ref_lon) * lon_to_meter + x_offset
        y_min = (building['min_lat'] - ref_lat) * lat_to_meter + y_offset
        x_max = (building['max_lon'] - ref_lon) * lon_to_meter + x_offset
        y_max = (building['max_lat'] - ref_lat) * lat_to_meter + y_offset
        
        local_buildings.append({
            'name': building['name'],
            'polygon': local_polygon,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        })
    
    return local_buildings

def visualize_buildings(buildings, start_coords, goal_coords, filename):
    """
    Create a visualization of the buildings
    
    Args:
        buildings: List of buildings with local coordinates
        start_coords: (x, y) of starting point
        goal_coords: (x, y) of goal point
        filename: Output filename for the image
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Set background color
    ax.set_facecolor('#f0f0f0')
    
    # Draw buildings as polygons
    for i, building in enumerate(buildings):
        # Draw the building outline
        poly = Polygon(building['polygon'], 
                       fill=True, 
                       color='red', 
                       alpha=0.5, 
                       linewidth=1, 
                       edgecolor='black')
        ax.add_patch(poly)
        
        # Draw the bounding box (as a dashed line)
        rect = Rectangle((building['x_min'], building['y_min']),
                         building['x_max'] - building['x_min'],
                         building['y_max'] - building['y_min'],
                         fill=False, linestyle='--', color='blue', alpha=0.3)
        ax.add_patch(rect)
        
        # Add building name for some buildings
        if i % 10 == 0:  # Label every 10th building to avoid clutter
            center_x = (building['x_min'] + building['x_max']) / 2
            center_y = (building['y_min'] + building['y_max']) / 2
            plt.text(center_x, center_y, building['name'], 
                     ha='center', va='center', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw start and goal points
    plt.scatter(start_coords[0], start_coords[1], s=100, color='green', marker='*', 
                label='Start (Engineering Hall)')
    plt.scatter(goal_coords[0], goal_coords[1], s=100, color='red', marker='*', 
                label='Goal (Chocolate Shoppe)')
    
    # Draw a straight line between start and goal (to visualize the direct path)
    plt.plot([start_coords[0], goal_coords[0]], [start_coords[1], goal_coords[1]], 
             color='blue', linestyle='--', alpha=0.7, label='Direct Path')
    
    # Set axis labels and title
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('UW-Madison Campus: Engineering Hall to Chocolate Shoppe Ice Cream')
    
    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    
    # Return the figure for display
    return fig

def save_to_csv(buildings, filename):
    """
    Save building data to CSV for use with CUDA program
    
    Args:
        buildings: List of buildings with local coordinates
        filename: Output CSV filename
    """
    # Create DataFrame with required columns
    df = pd.DataFrame([{
        'name': b['name'], 
        'x_min': b['x_min'], 
        'y_min': b['y_min'], 
        'x_max': b['x_max'], 
        'y_max': b['y_max']
    } for b in buildings])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Building data saved to {filename}")

def main():
    # Define coordinates for UW-Madison Engineering to Chocolate Shoppe area
    # Use the provided map boundary coordinates
    SOUTH = 43.06793  # Southern latitude boundary
    WEST = -89.41675  # Western longitude boundary 
    NORTH = 43.07743  # Northern latitude boundary
    EAST = -89.38992  # Eastern longitude boundary
    
    # Reference point (southwest corner of bounding box)
    REF_LAT = SOUTH
    REF_LON = WEST
    
    # Specific coordinates for Engineering Hall and Chocolate Shoppe
    ENG_HALL_LAT, ENG_HALL_LON = 43.072258, -89.410322  # Engineering Hall
    CHOC_SHOPPE_LAT, CHOC_SHOPPE_LON = 43.074850, -89.393180  # Chocolate Shoppe
    
    # Fetch data from OpenStreetMap
    data = fetch_buildings(SOUTH, WEST, NORTH, EAST)
    
    if data:
        # Extract building information
        buildings = extract_buildings(data)
        
        # Convert to local coordinates
        local_buildings = convert_to_local_coordinates(buildings, REF_LAT, REF_LON)
        
        # Calculate start and goal in local coordinates
        x_offset = 300  # Same offset used in conversion function
        y_offset = 300
        start_x = (ENG_HALL_LON - REF_LON) * 111320 * np.cos(np.radians(REF_LAT)) + x_offset
        start_y = (ENG_HALL_LAT - REF_LAT) * 111320 + y_offset
        
        goal_x = (CHOC_SHOPPE_LON - REF_LON) * 111320 * np.cos(np.radians(REF_LAT)) + x_offset
        goal_y = (CHOC_SHOPPE_LAT - REF_LAT) * 111320 + y_offset
        
        # Visualize the buildings
        visualize_buildings(local_buildings, (start_x, start_y), (goal_x, goal_y), 'campus_buildings.png')
        
        # Save data to CSV for use with CUDA program
        save_to_csv(local_buildings, 'building_obstacles.csv')
        
        # Print some info for configuring main2.cu
        print(f"\nCUDA Configuration Info:")
        print(f"START_X = {start_x:.1f}f;  // Engineering Hall")
        print(f"START_Y = {start_y:.1f}f;")
        print(f"GOAL_X = {goal_x:.1f}f;   // Chocolate Shoppe")
        print(f"GOAL_Y = {goal_y:.1f}f;")
        
        # Calculate world boundaries
        x_coords = [b['x_min'] for b in local_buildings] + [b['x_max'] for b in local_buildings]
        y_coords = [b['y_min'] for b in local_buildings] + [b['y_max'] for b in local_buildings]
        
        # Ensure start and goal are within bounds
        x_coords.extend([start_x, goal_x])
        y_coords.extend([start_y, goal_y])
        
        world_width = max(x_coords) * 1.1  # Add 10% margin
        world_height = max(y_coords) * 1.1
        
        print(f"WORLD_WIDTH = {world_width:.1f}f;  // With margin")
        print(f"WORLD_HEIGHT = {world_height:.1f}f;")
        
        print("\nDone! Building data has been fetched, visualized, and saved.")
    else:
        print("Failed to fetch building data.")

if __name__ == "__main__":
    main()