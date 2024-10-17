import requests
import csv

# List of species names
species_list = [
    "Neolamarckia cadamba", "Pongamia pinnata", "Alstonia scholaris", "Delonix regia",
    "Terminalia catappa", "Cassia fistula", "Albizia lebbeck", "Mimusops elengi",
    "Azadirachta indica", "Sorghum bicolor", "Mangifera indica", "Lannea coromandelica",
    "Peltophorum pterocarpum", "Solanum melongena", "Calotropis gigantea",
    "Oryza rufipogon", "Ficus benghalensis"
]

# GBIF API URL
gbif_api_url = "https://api.gbif.org/v1/species/match"

# Function to fetch taxonomy info for a species
def get_taxonomy_info(species_name):
    params = {
        "name": species_name,
        "rank": "species"
    }
    response = requests.get(gbif_api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('usageKey'):
            taxonomy_info = {
                "Scientific Name": data.get("scientificName", ""),
                "Kingdom": data.get("kingdom", ""),
                "Phylum": data.get("phylum", ""),
                "Class": data.get("class", ""),
                "Order": data.get("order", ""),
                "Family": data.get("family", ""),
                "Genus": data.get("genus", ""),
                "Species": data.get("species", ""),
                "Canonical Name": data.get("canonicalName", ""),
                "Authorship": data.get("authorship", ""),
                "Status": data.get("status", ""),
                "Rank": data.get("rank", ""),
                "Match Type": data.get("matchType", ""),
                "Basionym": data.get("basionym", "")
            }
            return taxonomy_info
    return {}

# List to store output data
output_data = []

# Fetch taxonomy information for each species
for species_name in species_list:
    taxonomy_info = get_taxonomy_info(species_name)
    taxonomy_info["Species Name"] = species_name  # Include species name in the result
    output_data.append(taxonomy_info)

# Write the output to a CSV file
output_file = "updated_species_taxonomy_info.csv"
with open(output_file, mode='w', newline='') as file:
    fieldnames = output_data[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

print(f"Updated taxonomy information saved to {output_file}")
