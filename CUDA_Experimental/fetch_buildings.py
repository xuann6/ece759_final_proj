import csv
import requests
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def fetch_buildings(south, west, north, east):
    """
    Fetch building data from OpenStreetMap using the Overpass API
    
    Args:
        south, west, north, east: Bounding box coordinates (latitude/longitude)
    
    Returns:
        List of buildings with coordinates
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
        return []
    
    data = response.json()
    
    # Process the response to extract building coordinates
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
                    'min_lat': min_lat,
                    'min_lon': min_lon,
                    'max_lat': max_lat,
                    'max_lon': max_lon
                })
    
    print(f"Found {len(buildings)} buildings")
    return buildings

def convert_to_local_coordinates(buildings, ref_lat, ref_lon, scale=1000.0):
    """
    Convert geographic coordinates to local coordinates
    
    Args:
        buildings: List of building dict with lat/lon coordinates
        ref_lat, ref_lon: Reference point (origin in local coordinates)
        scale: Scale factor to convert to meters
    
    Returns:
        List of buildings with local coordinates
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert to radians
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    local_buildings = []
    for building in buildings:
        # Calculate x distance (longitude)
        dx = np.radians(building['min_lon'] - ref_lon) * R * np.cos(ref_lat_rad)
        dx2 = np.radians(building['max_lon'] - ref_lon) * R * np.cos(ref_lat_rad)
        
        # Calculate y distance (latitude)
        dy = np.radians(building['min_lat'] - ref_lat) * R
        dy2 = np.radians(building['max_lat'] - ref_lat) * R
        
        # Convert to local scaled coordinates
        x_min = dx * scale / R
        y_min = dy * scale / R
        x_max = dx2 * scale / R
        y_max = dy2 * scale / R
        
        # Ensure x_min < x_max and y_min < y_max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
            
        # Adjust coordinates to be positive (shift if needed)
        # We add a shift to ensure values are in our chosen coordinate ranges
        x_shift = 100  # Shift to ensure positive coordinates within our range
        y_shift = 100
        
        local_buildings.append({
            'name': building['name'],
            'x_min': x_min + x_shift,
            'y_min': y_min + y_shift,
            'x_max': x_max + x_shift,
            'y_max': y_max + y_shift
        })
    
    return local_buildings

def save_buildings_to_csv(buildings, filename):
    """Save building data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['name', 'x_min', 'y_min', 'x_max', 'y_max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for building in buildings:
            writer.writerow(building)
    
    print(f"Saved {len(buildings)} buildings to {filename}")

def visualize_buildings(buildings, start_x, start_y, goal_x, goal_y, world_width, world_height, filename):
    """Create a visualization of the buildings and save it as an image"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw buildings
    for building in buildings:
        width = building['x_max'] - building['x_min']
        height = building['y_max'] - building['y_min']
        
        rect = Rectangle(
            (building['x_min'], building['y_min']),
            width, height,
            fill=True, color='red', alpha=0.5
        )
        ax.add_patch(rect)
        
        # Add building name if there's enough space
        if width > 20 and height > 20:
            plt.text(
                building['x_min'] + width/2,
                building['y_min'] + height/2,
                building['name'],
                ha='center', va='center',
                fontsize=8, color='black'
            )
    
    # Draw start and goal
    plt.scatter(start_x, start_y, s=100, color='green', marker='*', label='Start (Engineering Hall)')
    plt.scatter(goal_x, goal_y, s=100, color='red', marker='*', label='Goal (Chocolate Shoppe)')
    
    # Set plot limits
    plt.xlim(0, world_width)
    plt.ylim(0, world_height)
    plt.title('UW-Madison Campus - RRT Planning')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()

def main():
    # Define coordinates for the area containing Engineering Hall to Chocolate Shoppe
    # These coordinates are for UW-Madison campus and State Street area
    # Define the exact bounds you need based on the specific locations
    SOUTH = 43.070  # Southern latitude boundary
    WEST = -89.410  # Western longitude boundary
    NORTH = 43.080  # Northern latitude boundary
    EAST = -89.380  # Eastern longitude boundary
    
    # Reference point (used as origin in our local coordinate system)
    # This should be a point in the southwest corner of our area of interest
    REF_LAT = SOUTH
    REF_LON = WEST
    
    # Define world boundaries for the RRT algorithm
    WORLD_WIDTH = 1000.0
    WORLD_HEIGHT = 700.0
    
    # Define start and goal coordinates (Engineering Hall and Chocolate Shoppe)
    START_X = 200.0  # Engineering Hall
    START_Y = 350.0
    GOAL_X = 800.0   # Chocolate Shoppe
    GOAL_Y = 250.0
    
    # Fetch building data
    buildings = fetch_buildings(SOUTH, WEST, NORTH, EAST)
    
    # Convert to local coordinates
    local_buildings = convert_to_local_coordinates(buildings, REF_LAT, REF_LON)
    
    # Save to CSV
    save_buildings_to_csv(local_buildings, 'buildings.csv')
    
    # Create visualization
    visualize_buildings(
        local_buildings, 
        START_X, START_Y, 
        GOAL_X, GOAL_Y, 
        WORLD_WIDTH, WORLD_HEIGHT,
        'campus_map.png'
    )
    
    # Also save data for main2.cu in the format it expects
    save_buildings_to_csv(local_buildings, 'building_obstacles.csv')
    
    print("Done! Building data has been fetched, processed, and saved.")

if __name__ == "__main__":
    main()