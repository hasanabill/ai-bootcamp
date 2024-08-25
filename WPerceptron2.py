import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.WEIGHTS = np.zeros(input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        Z = self.WEIGHTS[0] + np.dot(X, self.WEIGHTS[1:])
        return np.array([self.activation_function(x) for x in Z])

    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)  # Insert bias term as first column
        for i in range(self.epochs):
            # Forward pass
            Z = np.dot(X, self.WEIGHTS)
            Y_PRED = np.array([self.activation_function(x) for x in Z])

            # Calculate error
            ERRORS = Y.flatten() - Y_PRED
            print(f"{i}   {ERRORS }")

            # Backword pass -- Update weights
            self.WEIGHTS = self.WEIGHTS + self.learning_rate * np.dot(X.T, ERRORS)

    def plot_decision_boundary(self, X, Y, perceptron, title):
        plt.figure(figsize=(8, 6))

        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.Spectral)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = np.array(
            [
                perceptron.predict(np.array([[x, y]]))
                for x, y in zip(np.ravel(xx), np.ravel(yy))
            ]
        )
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.show()


# OR gate data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y_train = np.array([[0], [1], [1], [1]])  # OR gate output

# Initialize and train the perceptron
perceptron = Perceptron(input_size=X_train.shape[1])  # 2 inputs
perceptron.fit(X_train, y_train)


# Plot decision boundary for training set
perceptron.plot_decision_boundary(
    X_train, y_train, perceptron, "Decision Boundary (Training Set)"
)


# OR gate data
X_test = np.array([[0, 1], [0, 0], [1, 1], [0, 0]])


# Predictions
print("X Test Predictions:")
for x in X_test:
    print(f"{x} -> {perceptron.predict(np.array([x]))}")
