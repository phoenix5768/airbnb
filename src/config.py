import os

dataset_path = os.path.join(os.getcwd(), '..', 'dataset')

REVIEWS_FILE = os.path.join(dataset_path, 'reviews.csv.gz')
LISTINGS_FILE = os.path.join(dataset_path, 'listings.csv.gz')
CALENDAR_FILE = os.path.join(dataset_path, 'calendar.csv.gz')
NEIGHBOURHOODS_FILE = os.path.join(dataset_path, 'neighbourhoods.csv')
NEIGHBOURHOODS_GEO_FILE = os.path.join(dataset_path, 'neighbourhoods.geojson')