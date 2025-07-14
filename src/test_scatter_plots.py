import data_cleaning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def train_and_get_predictions(X, y, model_type="gb"):
    """Train model and return predictions for both train and test sets."""
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "gb":
        model = GradientBoostingRegressor(
            n_estimators=201,
            learning_rate=0.0912,
            max_depth=5,
            random_state=42,
            subsample=0.7922
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:  # linear regression
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n{model_type.upper()} Results:")
    print(f"Training MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
    print(f"Testing MAE: {test_mae:.2f}, R²: {test_r2:.3f}")
    
    return y_train, train_pred, y_test, test_pred

def plot_scatter(y_true, y_pred, title, ax):
    """Create scatter plot with perfect prediction line."""
    # Plot scatter points
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate and plot regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'g-', label='Best Fit', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(title)
    ax.legend()
    
    # Add R^2 value
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, 
            verticalalignment='top')

# Load and prepare data
print("Loading data...")
listings = data_cleaning.get_listings_data()
reviews = data_cleaning.get_reviews_data()
calendar = data_cleaning.get_calendar_data()

# Prepare all features dataset
listings_with_reviews = listings.merge(reviews, how='left', left_on='id', right_on='listing_id')
listings_with_reviews['review_count'] = listings_with_reviews['review_count'].fillna(0)
listings_with_reviews['avg_review_length'] = listings_with_reviews['avg_review_length'].fillna(0)
listings_with_reviews['avg_sentiment'] = listings_with_reviews['avg_sentiment'].fillna(0)
listings_with_reviews['days_since_last_review'] = listings_with_reviews['days_since_last_review'].fillna(365)

listings_all = listings_with_reviews.merge(calendar, how='left', left_on='id', right_on='listing_id')
listings_all[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']] = \
    listings_all[['mean_price', 'median_price', 'price_std_dev', 'avg_availability']].fillna(0)

X_all, y = data_cleaning.get_final_data(listings_all)

# Get predictions for all models
y_train_gb, pred_train_gb, y_test_gb, pred_test_gb = train_and_get_predictions(X_all, y, "gb")
y_train_rf, pred_train_rf, y_test_rf, pred_test_rf = train_and_get_predictions(X_all, y, "rf")
y_train_lr, pred_train_lr, y_test_lr, pred_test_lr = train_and_get_predictions(X_all, y, "lr")

# Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Predicted vs Actual Prices by Model', fontsize=16, y=1.02)

# Training set plots
plot_scatter(y_train_gb, pred_train_gb, 'Gradient Boosting (Train)', axes[0, 0])
plot_scatter(y_train_rf, pred_train_rf, 'Random Forest (Train)', axes[0, 1])
plot_scatter(y_train_lr, pred_train_lr, 'Linear Regression (Train)', axes[0, 2])

# Test set plots
plot_scatter(y_test_gb, pred_test_gb, 'Gradient Boosting (Test)', axes[1, 0])
plot_scatter(y_test_rf, pred_test_rf, 'Random Forest (Test)', axes[1, 1])
plot_scatter(y_test_lr, pred_test_lr, 'Linear Regression (Test)', axes[1, 2])

# Adjust layout and display
plt.tight_layout()
plt.show()

# Create residual plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Residual Plots by Model', fontsize=16, y=1.02)

def plot_residuals(y_true, y_pred, title, ax):
    """Create residual plot."""
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Price')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    
    # Add standard deviation of residuals
    std_resid = np.std(residuals)
    ax.text(0.05, 0.95, f'Std Dev = {std_resid:.2f}', 
            transform=ax.transAxes, 
            verticalalignment='top')

# Training set residual plots
plot_residuals(y_train_gb, pred_train_gb, 'Gradient Boosting (Train)', axes[0, 0])
plot_residuals(y_train_rf, pred_train_rf, 'Random Forest (Train)', axes[0, 1])
plot_residuals(y_train_lr, pred_train_lr, 'Linear Regression (Train)', axes[0, 2])

# Test set residual plots
plot_residuals(y_test_gb, pred_test_gb, 'Gradient Boosting (Test)', axes[1, 0])
plot_residuals(y_test_rf, pred_test_rf, 'Random Forest (Test)', axes[1, 1])
plot_residuals(y_test_lr, pred_test_lr, 'Linear Regression (Test)', axes[1, 2])

# Adjust layout and display
plt.tight_layout()
plt.show()

# Create error distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Error Distribution by Model (Test Set)', fontsize=16, y=1.02)

def plot_error_dist(y_true, y_pred, title, ax):
    """Create error distribution plot."""
    errors = y_pred - y_true
    sns.histplot(errors, kde=True, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add mean and std
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.text(0.05, 0.95, f'Mean = {mean_err:.2f}\nStd = {std_err:.2f}', 
            transform=ax.transAxes, 
            verticalalignment='top')

# Test set error distributions
plot_error_dist(y_test_gb, pred_test_gb, 'Gradient Boosting', axes[0])
plot_error_dist(y_test_rf, pred_test_rf, 'Random Forest', axes[1])
plot_error_dist(y_test_lr, pred_test_lr, 'Linear Regression', axes[2])

# Adjust layout and display
plt.tight_layout()
plt.show() 