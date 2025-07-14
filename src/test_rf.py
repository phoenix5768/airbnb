import data_cleaning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd

def train_and_evaluate_rf(X, y, model_name="Model"):
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=False)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    print(f"\n{model_name} Results:")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE: {test_mae:.2f}")
    return train_mae, test_mae, model

print("Loading data...")
listings = data_cleaning.get_listings_data()
reviews = data_cleaning.get_reviews_data()
calendar = data_cleaning.get_calendar_data()

print("\nPreparing feature combinations...")
X_listings, y = data_cleaning.get_final_data(listings)
print(f"Listings features: {X_listings.shape[1]}")
listings_with_reviews = listings.merge(reviews, how='left', left_on='id', right_on='listing_id')
listings_with_reviews['review_count'] = listings_with_reviews['review_count'].fillna(0)
listings_with_reviews['avg_review_length'] = listings_with_reviews['avg_review_length'].fillna(0)
listings_with_reviews['avg_sentiment'] = listings_with_reviews['avg_sentiment'].fillna(0)
listings_with_reviews['days_since_last_review'] = listings_with_reviews['days_since_last_review'].fillna(365)
X_with_reviews, _ = data_cleaning.get_final_data(listings_with_reviews)
print(f"Listings + Reviews features: {X_with_reviews.shape[1]}")
listings_with_calendar = listings.merge(calendar, how='left', left_on='id', right_on='listing_id')
listings_with_calendar[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']] = \
    listings_with_calendar[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']].fillna(0)
X_with_calendar, _ = data_cleaning.get_final_data(listings_with_calendar)
print(f"Listings + Calendar features: {X_with_calendar.shape[1]}")
listings_all = listings_with_reviews.merge(calendar, how='left', left_on='id', right_on='listing_id')
X_all, _ = data_cleaning.get_final_data(listings_all)
print(f"All features: {X_all.shape[1]}")

results = {
    "Listings Only": train_and_evaluate_rf(X_listings, y, "Listings Only"),
    "With Reviews": train_and_evaluate_rf(X_with_reviews, y, "With Reviews"),
    "With Calendar": train_and_evaluate_rf(X_with_calendar, y, "With Calendar"),
    "All Features": train_and_evaluate_rf(X_all, y, "All Features")
}

# Bar plot for MAE
labels = list(results.keys())
train_maes = [results[k][0] for k in labels]
test_maes = [results[k][1] for k in labels]

x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(10,6))
plt.bar(x - width/2, train_maes, width, label='Train MAE')
plt.bar(x + width/2, test_maes, width, label='Test MAE')
plt.xticks(x, labels, rotation=15)
plt.ylabel('Mean Absolute Error')
plt.title('Random Forest MAE by Feature Set')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importances for best model
best_model_name = min(results.items(), key=lambda x: x[1][1])[0]
_, _, best_model = results[best_model_name]
X_best = {"Listings Only": X_listings, "With Reviews": X_with_reviews, 
          "With Calendar": X_with_calendar, "All Features": X_all}[best_model_name]
importances = best_model.feature_importances_
feature_names = X_best.columns
importance_pairs = list(zip(feature_names, importances))
importance_pairs.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 20 Most Important Features ({best_model_name}):")
for name, importance in importance_pairs[:20]:
    print(f"{name}: {importance:.4f}")
plt.figure(figsize=(12, 6))
top_n = 20
names = [pair[0] for pair in importance_pairs[:top_n]]
values = [pair[1] for pair in importance_pairs[:top_n]]
plt.barh(range(len(names)), values)
plt.yticks(range(len(names)), names)
plt.xlabel('Feature Importance')
plt.title(f'Top 20 Most Important Features ({best_model_name}) - RF')
plt.tight_layout()
plt.show() 