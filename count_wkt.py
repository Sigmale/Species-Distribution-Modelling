from shapely.wkt import loads
import sys 
def count_coordinates_from_wkt_file(wkt_file_path):
    """
    Read a WKT file and return the number of coordinates it contains.

    :param wkt_file_path: Path to the WKT file
    :return: Number of coordinates in the WKT geometry
    """
    try:
        with open(wkt_file_path, 'r') as file:
            wkt_data = file.read().strip()  # Read and strip whitespace

        # Load the geometry from the WKT
        geometry = loads(wkt_data)

        # Count the number of coordinates
        if geometry.is_empty:
            return 0  # Return 0 if the geometry is empty

        # Check if the geometry is a multi-geometry
        if geometry.geom_type.startswith("Multi"):
            num_coordinates = sum(len(geom.exterior.coords) for geom in geometry.geoms)
        else:
            num_coordinates = len(geometry.exterior.coords)

        return num_coordinates

    except Exception as e:
        print(f"Error reading WKT file: {e}")
        return None

# Example usage
wkt_file_path = sys.argv[1]  # Replace with your WKT file path
num_coords = count_coordinates_from_wkt_file(wkt_file_path)
if num_coords is not None:
    print(f"Number of coordinates: {num_coords}")
