import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

# Load the data (you can replace this with your own dataset)
data = pd.read_csv('simulation_data.csv')

# Feature Engineering

# Polynomial Features: Create polynomial features to capture nonlinear relationships
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(data[['feature1', 'feature2']])

# One-Hot Encoding: Convert categorical variables into binary vectors
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(data[['category']])

# Combine the engineered features with the original features
X_combined = pd.concat([pd.DataFrame(X_poly), pd.DataFrame(X_encoded.toarray())], axis=1)

# Now you can use the engineered features X_combined for modeling and simulation
