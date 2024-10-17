import requests
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# Load the WKT polygon for the eco-region
with open('eco_region_polygon.wkt', 'r') as file:
    wkt_polygon = file.read()

polygon = wkt.loads(wkt_polygon)

# GBIF API endpoint
url = "https://api.gbif.org/v1/occurrence/search"

def extract_worldclim_data(lon, lat, worldclim_files):
    point_data = []
    for file in worldclim_files:
        with rasterio.open(file) as src:
            try:
                val = list(src.sample([(lon, lat)]))[0][0]
                point_data.append(val)
            except IndexError:
                point_data.append(np.nan)
    return point_data

# Function to get presence data from GBIF
def get_presence_data(species_name, polygon):
    limit = 300
    offset = 0
    presence_points = []
    
    while True:
        params = {
            "scientificName": species_name,
            "geometry": wkt_polygon,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "limit": limit,
            "offset": offset,
            "kingdomKey": 6  # Filter for Plantae kingdom
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            for record in data['results']:
                lat, lon = record['decimalLatitude'], record['decimalLongitude']
                if polygon.contains(Point(lon, lat)):
                    presence_points.append((lon, lat))

            if len(data['results']) < limit:
                break
            
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break
    
    print(f"Total presence points for {species_name}: {len(presence_points)}")
    return presence_points

# Generate pseudo-absence points
def generate_pseudo_absence(polygon, n_points, presence_points):
    minx, miny, maxx, maxy = polygon.bounds
    pseudo_absence_points = []
    
    # Convert presence points to a set for faster lookup
    presence_set = set(presence_points)
    
    while len(pseudo_absence_points) < n_points:
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        
        point = (lon, lat)
        if polygon.contains(Point(lon, lat)) and point not in presence_set:
            pseudo_absence_points.append(point)
    
    return pseudo_absence_points

def calculate_reliability(point_features, scaled_presence_features, scaler):
    scaled_point_features = scaler.transform([point_features])
    distances = cdist(scaled_point_features, scaled_presence_features, metric='euclidean')
    similarities = np.exp(-distances**2 / (2 * scaled_presence_features.shape[1]))  # Using number of features as bandwidth
    similarity = np.mean(similarities)
    reliability = 1 - similarity
    return reliability

def generate_pseudo_absence_feature_similarity(presence_points, polygon, worldclim_files, n_points, reliability_threshold=0.99999):
    # Extract features for presence points
    presence_features = []
    for lon, lat in presence_points:
        features = extract_worldclim_data(lon, lat, worldclim_files)
        if not np.isnan(features).any():
            presence_features.append(features)
    
    presence_features = np.array(presence_features)

    # Scale features
    scaler = StandardScaler()
    scaled_presence_features = scaler.fit_transform(presence_features)

    # Generate pseudo-absence points
    pseudo_absence_points = []
    minx, miny, maxx, maxy = polygon.bounds
    
    pbar = tqdm(total=n_points, desc="Generating pseudo-absence points")
    while len(pseudo_absence_points) < n_points:
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        
        if polygon.contains(Point(lon, lat)):
            features = extract_worldclim_data(lon, lat, worldclim_files)
            if not np.isnan(features).any():
                reliability = calculate_reliability(features, scaled_presence_features, scaler)
                if reliability >= reliability_threshold:
                    pseudo_absence_points.append((lon, lat))
                    pbar.update(1)
    
    pbar.close()
    return pseudo_absence_points

# Function to predict probability of presence for the entire eco-region
def predict_eco_region(model, worldclim_files, polygon):
    print("Predicting probability of presence for the entire eco-region...")
    with rasterio.open(worldclim_files[0]) as src:
        bounds = polygon.bounds
        transform = from_bounds(*bounds, width=int((bounds[2]-bounds[0])/0.008333), height=int((bounds[3]-bounds[1])/0.008333))
        eco_region_mask = geometry_mask([polygon], transform=transform, out_shape=(int((bounds[3]-bounds[1])/0.008333), int((bounds[2]-bounds[0])/0.008333)), invert=True)
    
    rows, cols = eco_region_mask.shape
    features = []
    
    for i in tqdm(range(rows)):
        for j in range(cols):
            if eco_region_mask[i, j]:
                lon, lat = rasterio.transform.xy(transform, i, j)
                features.append(extract_worldclim_data(lon, lat, worldclim_files))
    
    X_pred = np.array(features)
    probabilities = model.predict_proba(X_pred)[:, 1]  # Probability of class 1 (presence)
    
    probability_map = np.zeros((rows, cols))
    k = 0
    for i in range(rows):
        for j in range(cols):
            if eco_region_mask[i, j]:
                probability_map[i, j] = probabilities[k]
                k += 1
    
    return probability_map, transform

# Function to plot feature histograms
def plot_feature_histograms(X, y, feature_names, save_dir="histogram_features"):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    presence = X[y == 1]
    absence = X[y == 0]
    
    num_features = X.shape[1]
    
    for i in range(num_features):
        plt.figure(figsize=(8, 4))
        plt.hist(presence[:, i], bins=30, alpha=0.5, label="Presence", color='blue')
        plt.hist(absence[:, i], bins=30, alpha=0.5, label="Absence", color='red')
        plt.title(f"Feature {i + 1} ({feature_names[i]})")
        plt.xlabel("Feature value")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        
        # Save the histogram as a PNG file
        filename = os.path.join(save_dir, f"feature_{i + 1}_{feature_names[i]}.png")
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory

    print(f"Histograms saved in '{save_dir}' directory.")
# Main execution
if __name__ == "__main__":
    # Set up WorldClim data
    worldclim_dir = "WorldClimBio"
    if not os.path.exists(worldclim_dir):
        print('WorldClim Data Not found')
        exit()   
    print('WorldClim Data Found....') 
    worldclim_files = [os.path.join(worldclim_dir, f) for f in os.listdir(worldclim_dir) if f.endswith('.tif')]

    # Get presence data for Alstonia scholaris
    presence_points = get_presence_data("Alstonia scholaris", polygon)
    csv_filename = 'presence.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['Longitude', 'Latitude'])
        
        # Write the data rows
        writer.writerows(presence_points)

    print(f"Data successfully written to {csv_filename}")
    # Generate pseudo-absence points using feature similarity
    print("Generating pseudo-absence points using feature similarity...")
    pseudo_absence_points = generate_pseudo_absence_feature_similarity(
        presence_points, polygon, worldclim_files, len(presence_points), reliability_threshold=0.7
    )

    # Combine presence and pseudo-absence points
    all_points = presence_points + pseudo_absence_points
    labels = [1] * len(presence_points) + [0] * len(pseudo_absence_points)

    all_points, labels = shuffle(all_points, labels, random_state=42)

    # Extract features for all points
    print("Extracting WorldClim data for all points...")
    features = []
    for lon, lat in tqdm(all_points):
        features.append(extract_worldclim_data(lon, lat, worldclim_files))

    # Prepare the dataset
    X = np.array(features)
    y = np.array(labels)

    # Remove any rows with NaN values
    valid_indices = ~np.isnan(X).any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    print("Training the Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate on the test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot and save feature histograms for presence vs pseudo-absence
    feature_names = [f"Feature {i + 1}" for i in range(X.shape[1])]
    print("Saving feature histograms to 'histogram_features' folder...")
    plot_feature_histograms(X, y, feature_names)