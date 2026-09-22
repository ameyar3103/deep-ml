import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int):

    updated_weights = initial_weights.copy()
    updated_b = initial_bias
    X = features
    y = labels
    mse_values = []
    m = len(features)

    for i in range(epochs):

        # sigmoid prediction
        z = X @ updated_weights + updated_b
        pred = 1 / (1 + np.exp(-z))

        # MSE BEFORE update
        mse_values.append(round(np.mean((pred - y) ** 2), 4))

        # gradient
        error = pred - y
        gradients_W = (2.0 / m) * (X.T @ (error * pred * (1 - pred)))
        gradients_b = (2.0 / m) * np.sum(error * pred * (1 - pred))

        # update
        updated_weights -= learning_rate * gradients_W
        updated_b -= learning_rate * gradients_b

    return np.round(updated_weights, 4), round(updated_b, 4), mse_values