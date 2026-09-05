import numpy as np
import pandas as pd

np.random.seed(42)
n = 545  # same size as the popular Kaggle Housing.csv

area = np.random.randint(1650, 16200, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
stories = np.random.randint(1, 5, n)
parking = np.random.randint(0, 4, n)

mainroad = np.random.choice(['yes', 'no'], n, p=[0.85, 0.15])
guestroom = np.random.choice(['yes', 'no'], n, p=[0.3, 0.7])
basement = np.random.choice(['yes', 'no'], n, p=[0.35, 0.65])
hotwaterheating = np.random.choice(['yes', 'no'], n, p=[0.1, 0.9])
airconditioning = np.random.choice(['yes', 'no'], n, p=[0.4, 0.6])
prefarea = np.random.choice(['yes', 'no'], n, p=[0.25, 0.75])
furnishingstatus = np.random.choice(
    ['furnished', 'semi-furnished', 'unfurnished'], n, p=[0.3, 0.4, 0.3]
)


price = (
    250000
    + area * 250
    + bedrooms * 150000
    + bathrooms * 300000
    + stories * 200000
    + parking * 100000
    + (mainroad == 'yes') * 350000
    + (guestroom == 'yes') * 150000
    + (basement == 'yes') * 200000
    + (hotwaterheating == 'yes') * 250000
    + (airconditioning == 'yes') * 400000
    + (prefarea == 'yes') * 300000
    + np.random.normal(0, 400000, n)  # noise
)
price = np.clip(price, 1750000, None).round(-3)

df = pd.DataFrame({
    'price': price.astype(int),
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus,
})

df.to_csv('data/Housing.csv', index=False)
print(df.head())
print(df.shape)
