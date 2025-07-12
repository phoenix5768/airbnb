import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def get_listings_data():
    df = pd.read_csv("../dataset/listings.csv.gz")

    # # Investigating data
    # logger.info("Shape:", df.shape)
    # logger.info(df.head())
    #
    # logger.info("Columns:", df.columns.tolist())

    # Cleaning data
    df['price'] = df['price'].replace('[\$,€]', '', regex=True).replace(',', '', regex=True).astype(float)
    df = df[df['price'] < 400]

    selected_cols = [
        'price',
        'room_type',
        'property_type',
        'accommodates',
        'bedrooms',
        'bathrooms',
        'beds',
        'latitude',
        'longitude',
        'neighbourhood_cleansed',
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

    ## Exploring missing data
    # missing = df.isnull().sum()
    # logger.info(missing[missing > 0].sort_values(ascending=False))

    # Numeric scores => fill with median
    for col in ['review_scores_rating', 'review_scores_value', 'review_scores_cleanliness', 'bedrooms', 'beds', 'bathrooms', 'host_total_listings_count']:
        df[col] = df[col].fillna(df[col].median())

    # reviews_per_month => fill with 0 (no reviews means zero per month)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    # host_is_superhost => fill missing with 'f' (not superhost)
    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')

    # Feature Engineering
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    df = pd.get_dummies(df, columns=['room_type', 'property_type', 'neighbourhood_cleansed'], drop_first=True)

    X = df.drop('price', axis=1)
    y = df['price']

    # Scaling numerical features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # logger.info(X)
    # logger.info(y)

    ## Exploring missing data
    # missing = df.isnull().sum()
    # logger.info(missing[missing > 0].sort_values(ascending=False))

    return X, y


def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    return text


def get_description():
    df = pd.read_csv("../dataset/listings.csv.gz")

    df['price'] = df['price'].replace('[\$,€]', '', regex=True).replace(',', '', regex=True).astype(float)
    df = df[df['price'] < 400]

    selected_cols = [
        'price',
        'room_type',
        'property_type',
        'accommodates',
        'bedrooms',
        'bathrooms',
        'beds',
        'latitude',
        'longitude',
        'neighbourhood_cleansed',
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
        'host_total_listings_count',
        'description'
    ]
    df = df[selected_cols]

    df['description'] = df['description']
    df['description_clean'] = df['description'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=100)  # limit to top 100 words to avoid overfitting
    text_features = tfidf.fit_transform(df['description_clean'])

    # Convert to DataFrame and reset index to align
    text_df = pd.DataFrame(text_features.toarray(), columns=tfidf.get_feature_names_out())
    text_df.reset_index(drop=True, inplace=True)

    return text_df