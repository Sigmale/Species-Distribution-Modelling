import json
from shapely.geometry import shape

def geojson_to_wkt(geojson_file, output_file):
    # Open the GeoJSON file and load it into a dictionary
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    
    # Convert GeoJSON to shapely geometry and then to WKT
    geometries = []
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])  # Convert GeoJSON to shapely geometry
        wkt = geom.wkt  # Convert shapely geometry to WKT
        geometries.append(wkt)
    
    # Save WKT geometries to output file in the requested format
    with open(output_file, 'w') as f:
        for wkt in geometries:
            f.write(f"{wkt}")

if __name__ == "__main__":
    geojson_file = 'North Western Ghats moist deciduous forests.geojson'  # Path to your GeoJSON file
    output_file = 'North Western Ghats moist deciduous forests.wkt'  # Path to save the WKT file
    
    geojson_to_wkt(geojson_file, output_file)
    print(f"WKT geometries saved to {output_file}")
