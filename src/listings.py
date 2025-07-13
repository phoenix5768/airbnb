import baseline_model
import data_cleaning


def main():
    listings = data_cleaning.get_listings_data()

    X, y = data_cleaning.get_final_data(listings)

    baseline_model.run_gb(X, y)
    baseline_model.run_linreg(X, y)
    baseline_model.run_rf(X, y)