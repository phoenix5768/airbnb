import data_cleaning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd

def train_and_evaluate_model(X, y, model_name="Model"):
    """Train model and return training curves and feature importances."""
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=201,
        learning_rate=0.0912,
        max_depth=5,
        random_state=42,
        subsample=0.7922
    )
    
    model.fit(X_train, y_train)
    
    # Calculate train and test errors at each iteration
    train_scores = []
    test_scores = []
    for pred_train, pred_test in zip(
        model.staged_predict(X_train),
        model.staged_predict(X_test)
    ):
        train_scores.append(mean_absolute_error(y_train, pred_train))
        test_scores.append(mean_absolute_error(y_test, pred_test))
    
    print(f"\n{model_name} Results:")
    print(f"Final Training MAE: {train_scores[-1]:.2f}")
    print(f"Final Testing MAE: {test_scores[-1]:.2f}")
    
    return train_scores, test_scores, model

# Load all data
print("Loading data...")
listings = data_cleaning.get_listings_data()
reviews = data_cleaning.get_reviews_data()
calendar = data_cleaning.get_calendar_data()

# Prepare different feature combinations
print("\nPreparing feature combinations...")

# Listings only
X_listings, y = data_cleaning.get_final_data(listings)
print(f"Listings features: {X_listings.shape[1]}")

# Listings + Reviews
listings_with_reviews = listings.merge(reviews, how='left', left_on='id', right_on='listing_id')
# Fill missing values as done in calendars.py
listings_with_reviews['review_count'] = listings_with_reviews['review_count'].fillna(0)
listings_with_reviews['avg_review_length'] = listings_with_reviews['avg_review_length'].fillna(0)
listings_with_reviews['avg_sentiment'] = listings_with_reviews['avg_sentiment'].fillna(0)
listings_with_reviews['days_since_last_review'] = listings_with_reviews['days_since_last_review'].fillna(365)
X_with_reviews, _ = data_cleaning.get_final_data(listings_with_reviews)
print(f"Listings + Reviews features: {X_with_reviews.shape[1]}")

# Listings + Calendar
listings_with_calendar = listings.merge(calendar, how='left', left_on='id', right_on='listing_id')
# Fill missing values as done in calendars.py
listings_with_calendar[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']] = \
    listings_with_calendar[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']].fillna(0)
X_with_calendar, _ = data_cleaning.get_final_data(listings_with_calendar)
print(f"Listings + Calendar features: {X_with_calendar.shape[1]}")

# All features
listings_all = listings_with_reviews.merge(calendar, how='left', left_on='id', right_on='listing_id')
# Calendar features already filled above
X_all, _ = data_cleaning.get_final_data(listings_all)
print(f"All features: {X_all.shape[1]}")

# Train models and get scores
results = {
    "Listings Only": train_and_evaluate_model(X_listings, y, "Listings Only"),
    "With Reviews": train_and_evaluate_model(X_with_reviews, y, "With Reviews"),
    "With Calendar": train_and_evaluate_model(X_with_calendar, y, "With Calendar"),
    "All Features": train_and_evaluate_model(X_all, y, "All Features")
}

# Plot training curves
plt.figure(figsize=(12, 6))
colors = ['blue', 'red', 'green', 'purple']
styles = ['-', '--', '-.', ':']

for (name, (train_scores, test_scores, _)), color, style in zip(results.items(), colors, styles):
    plt.plot(train_scores, label=f'{name} (Train)', color=color, linestyle=style, alpha=0.8)
    plt.plot(test_scores, label=f'{name} (Test)', color=color, linestyle=style, alpha=0.4)

plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Mean Absolute Error')
plt.title('Training Progress Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot feature importances for the best model
best_model_name = min(results.items(), key=lambda x: x[1][1][-1])[0]  # Model with lowest test MAE
_, _, best_model = results[best_model_name]
X_best = {"Listings Only": X_listings, "With Reviews": X_with_reviews, 
          "With Calendar": X_with_calendar, "All Features": X_all}[best_model_name]

importances = best_model.feature_importances_
feature_names = X_best.columns
importance_pairs = list(zip(feature_names, importances))
importance_pairs.sort(key=lambda x: x[1], reverse=True)

# Print top 20 most important features
print(f"\nTop 20 Most Important Features ({best_model_name}):")
for name, importance in importance_pairs[:20]:
    print(f"{name}: {importance:.4f}")

# Plot feature importances
plt.figure(figsize=(12, 6))
top_n = 20
names = [pair[0] for pair in importance_pairs[:top_n]]
values = [pair[1] for pair in importance_pairs[:top_n]]

plt.barh(range(len(names)), values)
plt.yticks(range(len(names)), names)
plt.xlabel('Feature Importance')
plt.title(f'Top 20 Most Important Features ({best_model_name})')
plt.tight_layout()
plt.show() 