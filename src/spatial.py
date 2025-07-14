import json
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from math import radians, cos, sin, asin, sqrt
from config import NEIGHBOURHOODS_FILE, NEIGHBOURHOODS_GEO_FILE

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Ensure all inputs are floats before converting to radians
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def add_neighborhood_clustering(df):
    """Add neighborhood clustering information to the dataframe."""
    try:
        # Load neighborhood mapping
        neigh_map = pd.read_csv(NEIGHBOURHOODS_FILE)
        print(f"Loaded neighborhood mapping with {len(neigh_map)} neighborhoods")
        
        # Merge to get neighborhood_group for each listing
        df = df.merge(neigh_map, left_on='neighbourhood_cleansed', right_on='neighbourhood', how='left')
        
        # Check if merge was successful
        if 'neighbourhood_group' in df.columns:
            print(f"Neighborhood clustering added - {df['neighbourhood_group'].nunique()} district groups")
            # Drop the redundant 'neighbourhood' column from the mapping
            if 'neighbourhood' in df.columns:
                df = df.drop('neighbourhood', axis=1)
        else:
            print("!!!!! Merge failed, using fallback")
            df['neighbourhood_group'] = df['neighbourhood_cleansed']
            
    except Exception as e:
        print(f"!!!! Could not load neighborhood clustering: {e}")
        # Create a fallback neighborhood_group column
        df['neighbourhood_group'] = df['neighbourhood_cleansed']
    
    return df

def load_neighborhood_geometries():
    """Load neighborhood geometries from GeoJSON file."""
    try:
        with open(NEIGHBOURHOODS_GEO_FILE, 'r') as f:
            geojson_data = json.load(f)
        
        neighborhoods = {}
        for feature in geojson_data['features']:
            name = feature['properties']['neighbourhood']
            geometry = feature['geometry']
            
            if geometry['type'] == 'MultiPolygon':
                polygons = []
                for coords in geometry['coordinates']:
                    for ring in coords:
                        polygon = Polygon(ring)
                        polygons.append(polygon)
                neighborhoods[name] = MultiPolygon(polygons)
            elif geometry['type'] == 'Polygon':
                neighborhoods[name] = Polygon(geometry['coordinates'][0])
        
        print(f"Loaded {len(neighborhoods)} neighborhood geometries from GeoJSON")
        return neighborhoods
    except Exception as e:
        print(f"Could not load neighborhood geometries: {e}")
        return {}

def calculate_neighborhood_spatial_features(df, neighborhoods):
    """Calculate spatial features based on neighborhood geometries."""
    if not neighborhoods:
        return df
    
    # Calculate neighborhood areas and centroids
    neighborhood_areas = {}
    neighborhood_centroids = {}
    
    for name, geometry in neighborhoods.items():
        # Calculate area in square kilometers
        area_km2 = geometry.area * 111 * 111  # Rough conversion from degrees to km²
        neighborhood_areas[name] = area_km2
        
        # Calculate centroid
        centroid = geometry.centroid
        neighborhood_centroids[name] = (centroid.y, centroid.x)  # lat, lon
    
    # Add features to dataframe
    df['neighborhood_area_km2'] = df['neighbourhood_cleansed'].map(neighborhood_areas)
    df['neighborhood_area_km2'] = df['neighborhood_area_km2'].fillna(df['neighborhood_area_km2'].median())
    
    # Distance to neighborhood centroid
    def distance_to_centroid(row):
        if row['neighbourhood_cleansed'] in neighborhood_centroids:
            centroid_lat, centroid_lon = neighborhood_centroids[row['neighbourhood_cleansed']]
            return haversine_distance(row['latitude'], row['longitude'], centroid_lat, centroid_lon)
        return 0
    
    df['distance_to_neighborhood_center_km'] = df.apply(distance_to_centroid, axis=1)
    
    # Neighborhood density (listing per km squared)
    neighborhood_counts = df['neighbourhood_cleansed'].value_counts()
    df['neighborhood_density'] = df['neighbourhood_cleansed'].map(neighborhood_counts) / df['neighborhood_area_km2']
    df['neighborhood_density'] = df['neighborhood_density'].fillna(df['neighborhood_density'].median())
    
    print(f"Added neighborhood spatial features: area, centroid distance, density")
    return df 