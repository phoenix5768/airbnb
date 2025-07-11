from sklearn.model_selection import train_test_split
from data_cleaning import get_listings_data, get_description
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def get_split_data():
    X, y = get_listings_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def linreg():
    X_train, X_test, y_train, y_test = get_split_data()

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    y_pred = model_lr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Linear Regression Results:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.3f}")


def random_forest():
    X_train, X_test, y_train, y_test = get_split_data()

    model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    logger.info(f"Random Forest Results:")
    logger.info(f"MAE: {mae_rf:.2f}")
    logger.info(f"RMSE: {rmse_rf:.2f}")
    logger.info(f"R² Score: {r2_rf:.3f}")


def rf_text():
    X, y = get_listings_data()
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


linreg()
random_forest()
rf_text()