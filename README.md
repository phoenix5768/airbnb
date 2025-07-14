# Airbnb Price Prediction - Madrid (SoSe 2025 ML Project)

## Authors: Batyrkhan Abukhanov & Sirazh Gabdullin

This project aims to predict nightly Airbnb listing prices in **Madrid** using rich, multimodal data from the [Inside Airbnb](http://insideairbnb.com/) dataset. The final model combines structured data, text data, temporal patterns, review sentiment, and location features to achieve strong prediction performance.

---

## Objective

Build a machine learning model that predicts the nightly price of an Airbnb listing in Madrid, based on structured listing data and possibly extended with other data types like text or spatial features.

---

## Project Structure
airbnb/
├── dataset/ # Raw datasets
├── src/ # Source code 
├── requirements.txt
└── README.md


---

## Dataset

Data is sourced from the [Inside Airbnb](http://insideairbnb.com/get-the-data.html) open-data project. The following files were used:

- `listings.csv.gz`: core features of each listing
- `reviews.csv.gz`: used for text sentiment or activity
- `calendar.csv.gz`: availability and pricing info
- `neighbourhoods.csv`: Spatial information about listing locations
- `neighbourhoods.geojson`: Spatial information about listing locations

---

## Data Preprocessing

- Filtered out listings priced above €200 to remove outliers.
- Selected relevant columns (room type, bedrooms, coordinates, reviews, etc.).
- Handled missing values using imputation or removal.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features using `StandardScaler`.

---

## Features Used
- **Structured features**: accommodates, bedrooms, bathrooms, review scores, host status, etc.
- **Text features**: description field vectorized using TF-IDF
- **Review features**: 
  - Average sentiment (TextBlob)
  - Review count
  - Average review length
  - Days since last review
- **Calendar features**:
  - Price dynamics
  - Availability ratio
- **Spatial features**: latitude, longitude 

---

## Models

We experimented with the following models:

-  **Linear Regression** (baseline)
-  **Random Forest Regressor**
-  ***Gradient Boosted Trees**

Evaluation Metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

---

## Results (Gradiendt Boosted Trees)

| Model               | MAE   | RMSE  | R² Score |
|--------------------|-------|-------|----------|
| **Gradient Boosting** | 17.77 | 23.61 | 0.724    |
| **Random Forest**     | 17.59 | 23.79 | 0.720    |
| **Linear Regression** | 21.73 | 27.99 | 0.613    |

Tree-based models clearly outperform linear regression, capturing nonlinear relationships and interactions among features.

---

## Tools & Libraries

- Python 3.x
- Pandas, NumPy, Scikit-learn
- Matplotlib / Seaborn for plots

Install dependencies:

```bash
pip install -r requirements.txt

The repo link:
```bash
https://github.com/phoenix5768/airbnb
