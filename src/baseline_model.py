from sklearn.model_selection import train_test_split
from listings import get_description
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


def rf_text(X, y):
    text_df = get_description()

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True)

    X_combined = pd.concat([X, text_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)
    logger.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    logger.info(f"R² Score: {r2_score(y_test, y_pred):.3f}")


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