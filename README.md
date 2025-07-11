# Airbnb Price Prediction - Berlin (SoSe 2025 ML Project)

## Authors
Batyrkhan Abukhanov & Sirazh Gabdullin

---

This project is part of the final assignment for the **Machine Learning course (SoSe 2025)**. The goal is to predict Airbnb nightly listing prices in **Berlin** using multiple data modalities such as structured listing features and potentially text or spatial data.

---

## Objective

Build a machine learning model that predicts the nightly price of an Airbnb listing in Berlin, based on structured listing data and possibly extended with other data types like text or spatial features.

---

## Dataset

Data is sourced from the [Inside Airbnb](http://insideairbnb.com/get-the-data.html) open-data project. The following files were used:

- `listings.csv`: core features of each listing
- (optional) `reviews.csv`: used for text sentiment or activity
- (optional) `calendar.csv`: availability and pricing info
- (optional) images or spatial data

---

## Data Preprocessing

- Filtered out listings priced above €400 to remove outliers.
- Selected relevant columns (room type, bedrooms, coordinates, reviews, etc.).
- Handled missing values using imputation or removal.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features using `StandardScaler`.

---

## Models

We experimented with the following models:

-  **Linear Regression** (baseline)
-  **Random Forest Regressor**
- (optional) Gradient Boosted Trees or Neural Networks

Evaluation Metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

---

## Results

| Model               | MAE (€) | RMSE (€) | R² Score |
|--------------------|---------|----------|----------|
| Linear Regression  | xx.xx   | xx.xx    | 0.xxx    |
| Random Forest      | xx.xx   | xx.xx    | 0.xxx    |

(*replace with actual results after training*)

---

## Tools & Libraries

- Python 3.x
- Pandas, NumPy, Scikit-learn
- Jupyter Notebook
- Matplotlib / Seaborn for plots

Install dependencies:

```bash
pip install -r requirements.txt
