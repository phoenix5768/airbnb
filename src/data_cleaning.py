import copy

import pandas as pd
import baseline_model
import re
import json
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from math import radians, cos, sin, asin, sqrt

from spatial import (
    load_neighborhood_geometries, 
    calculate_neighborhood_spatial_features, 
    haversine_distance,
    add_neighborhood_clustering
)

# Coordinates for Puerta del Sol, Madrid (city center)
CITY_CENTER_LAT = 40.4168
CITY_CENTER_LON = -3.7038

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    return text


def get_calendar_data():
    calendar = pd.read_csv('dataset/calendar.csv.gz')

    # Clean the price column
    calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)

    # Convert 'available' from 't'/'f' to 1/0
    calendar['available'] = calendar['available'].map({'t': 1, 'f': 0})

    price_stats = calendar.groupby('listing_id')['price'].agg(
        mean_price='mean',
        median_price='median',
        price_std_dev='std'
    ).reset_index()

    # Average availability
    availability_stats = calendar.groupby('listing_id')['available'].mean().reset_index()
    availability_stats.rename(columns={'available': 'avg_availability'}, inplace=True)

    calendar_features = price_stats.merge(availability_stats, on='listing_id')

    calendar_features[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']] = calendar_features[[
        'mean_price', 'median_price', 'price_std_dev', 'avg_availability'
    ]].fillna(0)

    return calendar_features


def get_listings_data():
    df = pd.read_csv("dataset/listings.csv.gz")
    df['price'] = df['price'].replace('[\$,€]', '', regex=True).replace(',', '', regex=True).astype(float)
    df = df[df['price'] < 400]

    # Compute distance to city center
    df['distance_to_center_km'] = df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], CITY_CENTER_LAT, CITY_CENTER_LON), axis=1
    )

    # Load neighborhood geometries from GeoJSON
    neighborhoods = load_neighborhood_geometries()
    
    # Add neighborhood clustering
    df = add_neighborhood_clustering(df)
    
    # Add advanced spatial features from GeoJSON
    df = calculate_neighborhood_spatial_features(df, neighborhoods)

    # Adding description
    desc_df = copy.deepcopy(df)
    desc_df['description_clean'] = desc_df['description'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=100)  # limit to top 100 words to avoid overfitting
    text_features = tfidf.fit_transform(desc_df['description_clean'])

    # Convert to DataFrame and reset index to align
    text_df = pd.DataFrame(text_features.toarray(), columns=tfidf.get_feature_names_out())
    text_df.reset_index(drop=True, inplace=True)
    text_df['id'] = desc_df['id'].values

    selected_cols = [
        'id',
        'price',
        'room_type',
        'property_type',
        'bedrooms',
        'bathrooms',
        'beds',
        'latitude',
        'longitude',
        'distance_to_center_km',
        'neighbourhood_cleansed',
        'neighbourhood_group',
        'neighborhood_area_km2',
        'distance_to_neighborhood_center_km',
        'neighborhood_density',
        'minimum_nights',
        'maximum_nights',
        'availability_30',
        'availability_365',
        'number_of_reviews',
        'reviews_per_month',
        'review_scores_rating',
        'review_scores_cleanliness',
        'review_scores_value',
        'host_is_superhost',
        'host_total_listings_count'
    ]

    df = df[selected_cols]

    df = df.merge(text_df, on='id', how='left')

    # Numeric scores => fill with median
    for col in ['review_scores_rating', 'review_scores_value', 'review_scores_cleanliness', 'bedrooms_x', 'beds', 'bathrooms', 'host_total_listings_count']:
        df[col] = df[col].fillna(df[col].median())

    # reviews_per_month => fill with 0 (no reviews means zero per month)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    # host_is_superhost => fill missing with 'f' (not superhost)
    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')

    # Feature Engineering
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    df = pd.get_dummies(df, columns=['room_type', 'property_type', 'neighbourhood_cleansed', 'neighbourhood_group'], drop_first=True)

    return df


def get_reviews_data():
    reviews = pd.read_csv("dataset/reviews.csv.gz")

    reviews['date'] = pd.to_datetime(reviews['date'])

    # Num of reviews
    review_counts = reviews.groupby('listing_id').size().rename('review_count')

    # Avg review length
    review_avg_len = reviews.groupby('listing_id')['comments'].apply(lambda x: x.dropna().str.len().mean()).rename(
        'avg_review_length')

    # Days since last review
    last_review_date = reviews.groupby('listing_id')['date'].max().rename('last_review_date')
    current_date = pd.to_datetime('today')
    days_since_last_review = (current_date - last_review_date).dt.days.rename('days_since_last_review')

    # Sentiment analysis
    reviews['sentiment'] = reviews['comments'].apply(baseline_model.get_sentiment)

    # Average sentiment per listing
    avg_sentiment = reviews.groupby('listing_id')['sentiment'].mean().rename('avg_sentiment')

    # Combine all
    review_features = pd.concat([review_counts, review_avg_len, days_since_last_review, avg_sentiment], axis=1)

    review_features['review_count'] = review_features['review_count'].fillna(0)
    review_features['avg_review_length'] = review_features['avg_review_length'].fillna(0)
    review_features['avg_sentiment'] = review_features['avg_sentiment'].fillna(0)

    review_features['days_since_last_review'] = review_features['days_since_last_review'].fillna(365)

    return review_features


def get_final_data(df):
    df = df.drop('id', axis=1)
    X = df.drop('price', axis=1)
    y = df['price']


    # Scaling numerical features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y