import baseline_model
import data_cleaning


def main():
    listings = data_cleaning.get_listings_data()
    reviews = data_cleaning.get_reviews_data()

    listings = listings.merge(reviews, how='left', left_on='id', right_on='listing_id')

    listings['review_count'] = listings['review_count'].fillna(0)
    listings['avg_review_length'] = listings['avg_review_length'].fillna(0)
    listings['avg_sentiment'] = listings['avg_sentiment'].fillna(0)
    listings['days_since_last_review'] = listings['days_since_last_review'].fillna(365)


    X, y = data_cleaning.get_final_data(listings)

    baseline_model.run_gb(X, y)
    baseline_model.run_linreg(X, y)
    baseline_model.run_rf(X, y)


main()