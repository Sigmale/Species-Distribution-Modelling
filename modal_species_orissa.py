import requests
from shapely import wkt
from shapely.geometry import Polygon

# Load the WKT polygon from the file (North_Western_Ghats_moist_deciduous_forests.wkt)
with open('North_Western_Ghats_moist_deciduous_forests.wkt', 'r') as file:
    wkt_polygon = file.read()

# Parse the WKT polygon using Shapely
polygon = wkt.loads(wkt_polygon)

# Simplify the polygon with a tolerance (higher values = more simplification)
tolerance = 0.1  # Adjust this value as needed
simplified_polygon = polygon.simplify(tolerance, preserve_topology=True)

# Convert the simplified polygon back to WKT format
simplified_wkt_polygon = simplified_polygon.wkt

# GBIF API endpoint
url = "https://api.gbif.org/v1/occurrence/search"

# Initialize variables for pagination
limit = 300  # GBIF API default is 300 records per request
offset = 0  # Start from the first record
species_count = {}
total_records = 0
Genus = {}
Genus_Count = {}

# Function to process each batch of records
def process_batch(data):
    global total_records
    for record in data['results']:
        species = record.get('species')
        genus = record.get('genus')
        if genus in Genus_Count:
            Genus_Count[genus] += 1
        else:
            Genus_Count[genus] = 1
        if species in species_count:
            species_count[species] += 1
            Genus[species] = genus
        else:
            species_count[species] = 1
            Genus[species] = genus
            print(f'discovered {species}')
        total_records += 1

# Continue fetching records until no more results are returned
while True:
    # Parameters for the GBIF API request, using the simplified WKT polygon and pagination
    params = {
        "geometry": simplified_wkt_polygon,  # Simplified polygon geometry
        "rank": "SPECIES",  # Query species-level data
        "limit": limit,  # Number of records to fetch in each request
        "offset": offset,  # The starting point for the next batch of records
        "hasCoordinate": "true",  # Ensure records have coordinates
        "hasGeospatialIssue": "false",  # Exclude records with geospatial issues
        "kingdomKey": 6,  # Filter for Plantae kingdom (key 6 in GBIF)
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Process the current batch of records
        process_batch(data)

        # If fewer results are returned than the limit, we're done
        if len(data['results']) < limit:
            break

        # Increment the offset to fetch the next batch of records
        offset += limit
    else:
        print(f"Error: {response.status_code}")
        break

# Sort species by occurrence counts in descending order
sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)

# Write sorted unique species and their counts to a text file
with open('unique_tree_species.txt', 'w') as file:
    file.write("Species # Genus # Occurrences\n")  # Header
    for species, count in sorted_species:
        file.write(f"{species} # {count} # {Genus[species]} # {Genus_Count[Genus[species]]}\n")

# Print the total number of unique species and total records fetched
print(f"Total number of unique species: {len(species_count)}")
print(f"Total records processed: {total_records}")
print("Unique tree species and their occurrences have been written to 'unique_tree_species.txt'.")
