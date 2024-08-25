import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(
        self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=20000
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Xavier Initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2.0 / (self.input_size + self.hidden_size)
        )
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2.0 / (self.hidden_size + self.output_size)
        )
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1.0 - np.tanh(z) ** 2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.tanh(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        return a2

    def fit(self, X, y):
        y_one_hot = np.eye(self.output_size)[y.reshape(-1)]  # One-hot encoding

        for epoch in range(self.epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.tanh(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.softmax(z2)

            # Compute error
            error = y_one_hot - a2
            d_a2 = error

            error_hidden = np.dot(d_a2, self.W2.T)
            d_a1 = error_hidden * self.tanh_derivative(a1)

            # Update weights and biases
            self.W2 += np.dot(a1.T, d_a2) * self.learning_rate
            self.b2 += np.sum(d_a2, axis=0, keepdims=True) * self.learning_rate
            self.W1 += np.dot(X.T, d_a1) * self.learning_rate
            self.b1 += np.sum(d_a1, axis=0, keepdims=True) * self.learning_rate

            if epoch % 1000 == 0:
                loss = (
                    -np.sum(y_one_hot * np.log(a2 + 1e-9)) / y_one_hot.shape[0]
                )  # Cross-entropy loss
                print(f"Epoch {epoch}, Loss: {loss}")


# XOR gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR gate output

# Convert y to one-hot encoding for Softmax
y_one_hot = np.eye(2)[y].reshape(-1, 2)

# Initialize and train the MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=2, learning_rate=0.01, epochs=20000)
mlp.fit(X, y)

# Predictions
print("Predictions:")
for x in X:
    print(f"{x} -> {mlp.predict(x)}")


# Plot decision boundary
def plot_decision_boundary(X, y, mlp, title):
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = np.array(
        [
            mlp.predict(np.array([[x, y]]))[:, 1]
            for x, y in zip(np.ravel(xx), np.ravel(yy))
        ]
    )
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()


# Plot decision boundary for training set
plot_decision_boundary(X, y, mlp, "Decision Boundary (Training Set)")
