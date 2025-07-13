import data_cleaning
import baseline_model


def main():
    listings = data_cleaning.get_listings_data()
    reviews = data_cleaning.get_reviews_data()
    calendars = data_cleaning.get_calendar_data()

    df = listings.merge(reviews, how='left', left_on='id', right_on='listing_id')
    df = df.merge(calendars, how='left', left_on='id', right_on='listing_id')


    df['review_count'] = df['review_count'].fillna(0)
    df['avg_review_length'] = df['avg_review_length'].fillna(0)
    df['avg_sentiment'] = df['avg_sentiment'].fillna(0)
    df['days_since_last_review'] = df['days_since_last_review'].fillna(365)
    df[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']] = df[[
        'mean_price', 'median_price', 'price_std_dev', 'avg_availability'
    ]].fillna(0)

    X, y = data_cleaning.get_final_data(listings)

    baseline_model.run_gb(X, y)
    baseline_model.run_linreg(X, y)
    baseline_model.run_rf(X, y)


main()