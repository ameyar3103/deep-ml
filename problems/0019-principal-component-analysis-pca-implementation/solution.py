import numpy as np
from sklearn.preprocessing import StandardScaler

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Returns:
        Array of shape (n_features, k), rounded to 4 decimals.
    """
    X = StandardScaler().fit_transform(data)

    cov = (X.T @ X) / (X.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]

    components = eigenvectors[:, :k]

    for j in range(k):
        for i in range(components.shape[0]):
            if abs(components[i, j]) > 1e-10:
                if components[i, j] < 0:
                    components[:, j] *= -1
                break

    return np.round(components, 4)