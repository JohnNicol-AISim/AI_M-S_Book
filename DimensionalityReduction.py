import numpy as np
from sklearn.decomposition import PCA

# Sample simulation data (replace this with your own dataset)
data = np.random.rand(100, 10)

# Dimensionality Reduction with PCA
num_components = 5  # Number of components to retain after dimensionality reduction
pca = PCA(n_components=num_components)
reduced_data = pca.fit_transform(data)

# Now you can use the reduced_data for your simulation, which has fewer dimensions
