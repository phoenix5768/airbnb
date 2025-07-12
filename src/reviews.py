import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import baseline_model


def get_sentiment(text):
    if pd.isnull(text):
        return 0
    return TextBlob(text).sentiment.polarity  # Returns value in [-1.0, 1.0]


def get_cleaned_listings(data):
    data['price'] = data['price'].replace('[\$,€]', '', regex=True).replace(',', '', regex=True).astype(float)
    data = data[data['price'] < 400]

    selected_cols = [
        'price',
        'review_count',
        'avg_review_length',
        'days_since_last_review',
        'avg_sentiment',
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
    data = data[selected_cols]

    # Numeric scores => fill with median
    for col in ['review_scores_rating', 'review_scores_value', 'review_scores_cleanliness', 'bedrooms', 'beds', 'bathrooms', 'host_total_listings_count']:
        data[col] = data[col].fillna(data[col].median())

    # reviews_per_month => fill with 0 (no reviews means zero per month)
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
    data['review_count'] = data['review_count'].fillna(0)
    data['avg_review_length'] = data['avg_review_length'].fillna(0)
    data['avg_sentiment'] = data['avg_sentiment'].fillna(0)

    data['days_since_last_review'] = data['days_since_last_review'].fillna(365)

    # host_is_superhost => fill missing with 'f' (not superhost)
    data['host_is_superhost'] = data['host_is_superhost'].fillna('f')

    # Feature Engineering
    data['host_is_superhost'] = data['host_is_superhost'].map({'t': 1, 'f': 0})
    data = pd.get_dummies(data, columns=['room_type', 'property_type', 'neighbourhood_cleansed'], drop_first=True)

    X = data.drop('price', axis=1)
    y = data['price']

    # Scaling numerical features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


def get_prepared_data():
    reviews = pd.read_csv("../dataset/reviews.csv.gz")
    listings = pd.read_csv("../dataset/listings.csv.gz")

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
    reviews['sentiment'] = reviews['comments'].apply(get_sentiment)

    # Average sentiment per listing
    avg_sentiment = reviews.groupby('listing_id')['sentiment'].mean().rename('avg_sentiment')

    # Combine all
    review_features = pd.concat([review_counts, review_avg_len, days_since_last_review, avg_sentiment], axis=1)

    # Merge with listings
    listings = listings.merge(review_features, how='left', left_on='id', right_on='listing_id')
    # listings = listings.join(review_features, how='left')
    listings['avg_sentiment'] = listings['avg_sentiment'].fillna(0)

    X, y = get_cleaned_listings(listings)

    return X, y



# baseline_model.hyperp_tune(X, y)
X, y = get_prepared_data()
baseline_model.run_gb(X, y)
# baseline_model.run_linreg(X, y)
