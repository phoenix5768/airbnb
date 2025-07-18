from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


def run_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest Results:")
    print(f"MAE: {mae_rf:.2f}")
    print(f"RMSE: {rmse_rf:.2f}")
    print(f"R² Score: {r2_rf:.3f}")

    # # Print feature importances
    # importances = model_rf.feature_importances_
    # feature_names = X.columns
    # sorted_idx = importances.argsort()[::-1]
    # print("Top 10 Random Forest Feature Importances:")
    # for i in range(min(10, len(feature_names))):
    #     print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")
    #
    # # Print full feature importance list
    # print("\nFull Random Forest Feature Importances:")
    # for i in range(len(feature_names)):
    #     print(f"{i+1:2d}. {feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")


def run_gb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    gbr = GradientBoostingRegressor(
        n_estimators=201,
        learning_rate=0.0912,
        max_depth=5,
        random_state=42,
        subsample=0.7922
    )

    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Gradient Boosting Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    feature_importance = gbr.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_n = 15
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance[sorted_idx][:top_n], y=np.array(X.columns)[sorted_idx][:top_n])
    plt.title("Top 15 Feature Importances (Gradient Boosting)")
    plt.tight_layout()
    plt.show()

    # # Print top 10 features
    # print("Top 10 Gradient Boosting Feature Importances:")
    # for i in range(min(10, len(X.columns))):
    #     print(f"{X.columns[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")
    #
    # # Print full feature importance list
    # print("\nFull Gradient Boosting Feature Importances:")
    # for i in range(len(X.columns)):
    #     print(f"{i+1:2d}. {X.columns[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")


def run_linreg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # # Print top 10 coefficients
    # coefs = lr.coef_
    # feature_names = X.columns
    # sorted_idx = np.abs(coefs).argsort()[::-1]
    # print("Top 10 Linear Regression Coefficients:")
    # for i in range(min(10, len(feature_names))):
    #     print(f"{feature_names[sorted_idx[i]]}: {coefs[sorted_idx[i]]:.4f}")
    #
    # # Print full coefficient list
    # print("\nFull Linear Regression Coefficients:")
    # for i in range(len(feature_names)):
    #     print(f"{i+1:2d}. {feature_names[sorted_idx[i]]}: {coefs[sorted_idx[i]]:.4f}")


def hyperp_tune(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 6),
        'subsample': uniform(0.7, 0.3)
    }

    random_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(best_model)


def get_sentiment(text):
    if pd.isnull(text):
        return 0
    return TextBlob(text).sentiment.polarity  # Returns value in [-1.0, 1.0]