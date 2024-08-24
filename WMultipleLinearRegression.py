import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data for demonstration
np.random.seed(0)
sqft = np.random.randint(1000, 3000, 1000)  # Sample square footage
bedrooms = np.random.randint(1, 5, 1000)  # Sample number of bedrooms
price = (
    50 * sqft + 10000 * bedrooms + np.random.normal(0, 5000, 1000)
)  # Sample price (with some noise)

sqft_train = sqft
bedrooms_train = bedrooms
price_train = price

# Reshape the data to be column vectors
sqft = sqft.reshape(-1, 1)
bedrooms = bedrooms.reshape(-1, 1)
price = price.reshape(-1, 1)

# Normalize the features (optional but recommended for gradient descent)
sqft_mean = np.mean(sqft)
sqft_std = np.std(sqft)
sqft_normalized = (sqft - sqft_mean) / sqft_std

bedrooms_mean = np.mean(bedrooms)
bedrooms_std = np.std(bedrooms)
bedrooms_normalized = (bedrooms - bedrooms_mean) / bedrooms_std

# Add a column of ones for the bias term
X = np.hstack([np.ones_like(sqft_normalized), sqft_normalized, bedrooms_normalized])

# Initialize weights randomly
np.random.seed(42)
weights = np.random.randn(3, 1)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 10000

# Gradient Descent
for i in range(num_iterations):
    # Compute predictions
    predictions = np.dot(X, weights)

    # Compute error
    error = predictions - price

    # Compute gradients
    gradients = 1 / len(price) * np.dot(X.T, error)

    # Update weights
    weights = weights - learning_rate * gradients

# Denormalize weights
intercept = (
    weights[0][0]
    - weights[1][0] * sqft_mean / sqft_std
    - weights[2][0] * bedrooms_mean / bedrooms_std
)
slope_sqft = weights[1][0] / sqft_std
slope_bedrooms = weights[2][0] / bedrooms_std

# Plot the data and regression plane (projected on sqft and price)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(sqft_train, bedrooms_train, price_train, color="blue", label="Data")

# Create a grid for sqft and bedrooms
sqft_grid, bedrooms_grid = np.meshgrid(
    np.linspace(sqft.min(), sqft.max(), 10),
    np.linspace(bedrooms.min(), bedrooms.max(), 10),
)
price_grid = intercept + slope_sqft * sqft_grid + slope_bedrooms * bedrooms_grid

# Plot the regression plane
ax.plot_surface(sqft_grid, bedrooms_grid, price_grid, color="red", alpha=0.5)

ax.set_title("Square Footage, Bedrooms vs. Price")
ax.set_xlabel("Square Footage")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")
ax.legend()
plt.show()

# Predict prices for some new square footages and bedrooms
new_sqft = np.array([[1500], [2000], [2500]])
new_bedrooms = np.array([[2], [3], [4]])

# Calculate predicted prices using the learned model
predicted_prices = intercept + slope_sqft * new_sqft + slope_bedrooms * new_bedrooms

print("Predicted prices:")
for i in range(len(new_sqft)):
    print(
        f"Square Footage: {new_sqft[i][0]}, Bedrooms: {new_bedrooms[i][0]}, Predicted Price: {predicted_prices[i][0]:.2f}"
    )

# Print the equation of the regression plane
print(
    f"Regression equation: Price = {intercept:.2f} + {slope_sqft:.2f} * sqft + {slope_bedrooms:.2f} * bedrooms"
)
