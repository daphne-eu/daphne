# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api.python.context.daphne_context import DaphneContext
import pandas as pd
import numpy as np
import torch
import time

def kmeans_pytorch(X, n_clusters, max_iters=1000, tol=1e-4):
    """
    Performs k-means clustering using PyTorch.
    
    Parameters:
    - X: The data tensor.
    - n_clusters: Number of clusters.
    - max_iters: Maximum number of iterations.
    - tol: Tolerance for convergence.
    
    Returns:
    - centroids: Cluster centers.
    - labels: Data labels.
    """
    # Initialize cluster centroids randomly from the data
    n_samples, n_features = X.size()
    init_indices = torch.randint(0, n_samples, (n_clusters,))
    centroids = X[init_indices]
    labels = torch.empty((n_samples,), dtype=torch.long)
    
    # Iteratively update centroids and labels
    for _ in range(max_iters):
        dist_matrix = torch.norm(X[:, None] - centroids, dim=2)
        labels = torch.argmin(dist_matrix, dim=1)
        new_centroids = torch.stack([X[labels == i].mean(0) for i in range(n_clusters)])
        
        # Break the loop if convergence is achieved
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
        
    return centroids, labels


print("\n\n###\n### Real World Test (Iris Dataset) Daphne - Python\n###\n")

# 1. Data Loading

print("1. Data Loading...")

# Load the Iris dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=column_names)

# Convert categorical class labels to numerical for ease of computation
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['class'] = df['class'].map(class_mapping)

dc = DaphneContext()

# Load the Data into Daphne
F = dc.from_pandas(df)
df = F.compute()

# Split data into features (X) and target (y)
#print(df.iloc[:, :-1].values)
X = dc.from_numpy(df.iloc[:, :-1].values)
X.compute()
y = dc.from_numpy(df.iloc[:, -1].values)
y.compute()


print("Data successfully loaded and class labels encoded.")
print("First 5 rows of the dataset:")
print(df.head())
print("\n")


# 2. Data Preprocessing

print("2. Data Preprocessing...")

start_preprocessing = time.time()

# Handle missing values by replacing them with mean
X.replace(np.nan, X.mean()).compute()

# Normalize the feature data to mean=0 and standard deviation=1
X = (X - X.mean(1)) / X.stddev(1)
X.compute()

end_preprocessing = time.time() - start_preprocessing

# Convert the Processed Matrix back into a DF for easier display
df = pd.DataFrame(data=np.append(X.compute(), y.compute(), axis=1), columns= column_names)
df['class'] = df['class'].astype(int)


print(f"Data preprocessing completed in {end_preprocessing:.4f} seconds.")
print("First 5 rows after preprocessing:")
print(df.head())
print("\n")


# 3. Matrix Operations - Not working with Native Daphne Commands

print("3. Matrix Operations...")

start_matrix_operations = time.time()


cov_matrix = np.cov(X.compute(), rowvar=False)

# Perform eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

end_matrix_operations = time.time() - start_matrix_operations
print(f"Covariance matrix computed and eigen decomposition done in {end_matrix_operations:.4f} seconds.")
print("Eigenvalues of the covariance matrix:")
print(eigenvalues)
print("\n")


# 4. Data Science Application

print("4. Data Science Application (k-means clustering)...")

start_application = time.time()

# Convert Daphne Matrix to torch tensor for PyTorch operations
X_tensor = X.compute(isPytorch=True)

# Apply k-means clustering
centroids, labels = kmeans_pytorch(X_tensor, n_clusters=3)

end_application = time.time() - start_application
print(f"k-means clustering applied in {end_application:.4f} seconds.")
print("Cluster centroids:")
print(centroids)
print("\n")

# 5. Linear Regression

print("5. Linear Regression...")

start_regression = time.time()

# Add a bias term (intercept) to the feature data
X = X.cbind(dc.fill(1.0, X.nrow(), 1))

# Calculate linear regression coefficients using ridge regression formula
lambda_ = dc.fill(0.001, X.ncol(), 1)

A = X.t() @ X + lambda_.diagMatrix()
b = X.t() @ y
beta = A.solve(b).compute()

end_regression = time.time() - start_regression
print(f"Linear regression successfully applied in {end_regression:.4f} seconds.")
print("Regression coefficients:")
print(beta)
print("\n")

# Create a Dataframe to Show the Execution Times

daphne_times = {
    "Preprocessing": end_preprocessing,
    "Matrix Operations": end_matrix_operations,
    "Application": end_application,
    "Regression": end_regression
}

times_df = pd.DataFrame({
    "Daphne Methods in Python": daphne_times,
})

print("Execution Times (in seconds):")
print(times_df)
