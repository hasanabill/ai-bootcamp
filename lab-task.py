import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

engine_size = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 100)
horsepower = np.random.randint(70, 400, 100)
vehicle_age = np.random.randint(1, 20, 100)
num_doors = np.random.randint(2, 5, 100)
price = (
    100 * engine_size
    + 500 * horsepower
    + 10 * vehicle_age
    + 5 * num_doors
    + np.random.normal(0, 5000, 100)
)

# Scale and shift the price to range from 10000 to 50000
price = 10000 + (price - np.min(price)) * (50000 - 10000) / (
    np.max(price) - np.min(price)
)

# print(engine_size)

engine_size_train = engine_size
horsepower_train = horsepower
vehicle_age_train = vehicle_age
num_doors_train = num_doors
price_train = price


# Reshape the data to be column vectors
engine_size = engine_size.reshape(-1, 1)
horsepower = horsepower.reshape(-1, 1)
vehicle_age = vehicle_age.reshape(-1, 1)
num_doors = num_doors.reshape(-1, 1)
price = price.reshape(-1, 1)


# Normalize the features
def normalize(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    normalized = (feature - mean) / std
    return normalized, mean, std


engine_size_normalized, engine_size_mean, engine_size_std = normalize(engine_size)
horsepower_normalized, horsepower_mean, horsepower_std = normalize(horsepower)
vehicle_age_normalized, vehicle_age_mean, vehicle_age_std = normalize(vehicle_age)
num_doors_normalized, num_doors_mean, num_doors_std = normalize(num_doors)

# Combine normalized features into a new feature matrix
X_normalized = np.hstack(
    [
        np.ones_like(engine_size_normalized),
        engine_size_normalized,
        horsepower_normalized,
        vehicle_age_normalized,
        num_doors_normalized,
    ]
)
# Initialize weights randomly
np.random.seed(42)
weights = np.random.randn(5, 1)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 10000

# Gradient Descent
for i in range(num_iterations):
    predictions = np.dot(X_normalized, weights)
    error = predictions - price.reshape(-1, 1)
    gradients = 1 / len(price) * np.dot(X_normalized.T, error)
    weights = weights - learning_rate * gradients

# Denormalize weights
intercept = (
    weights[0][0]
    - weights[1][0] * engine_size_mean / engine_size_std
    - weights[2][0] * horsepower_mean / horsepower_std
    - weights[3][0] * vehicle_age_mean / vehicle_age_std
    - weights[4][0] * num_doors_mean / num_doors_std
)
slope_engine_size = weights[1][0] / engine_size_std
slope_horsepower = weights[2][0] / horsepower_std
slope_vehicle_age = weights[3][0] / vehicle_age_std
slope_num_doors = weights[4][0] / num_doors_std

# Regression Plane
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(engine_size_train, horsepower_train, price_train, color="blue", label="Data")

# grid for Engine Size and Horsepower
engine_size_grid, horsepower_grid = np.meshgrid(
    np.linspace(engine_size.min(), engine_size.max(), 10),
    np.linspace(horsepower.min(), horsepower.max(), 10),
)
price_grid = (
    intercept
    + slope_engine_size * engine_size_grid
    + slope_horsepower * horsepower_grid
)

# Plot the regression plane
ax.plot_surface(engine_size_grid, horsepower_grid, price_grid, color="red", alpha=0.5)

ax.set_title("Engine Size, Horsepower vs. Price")
ax.set_xlabel("Engine Size (liters)")
ax.set_ylabel("Horsepower")
ax.set_zlabel("Price ($)")
ax.legend()
plt.show()

# Predict Prices for New Data
new_engine_size = np.array([[2.0], [3.5], [4.0]])
new_horsepower = np.array([[150], [250], [320]])
new_vehicle_age = np.array([[5], [3], [7]])
new_num_doors = np.array([[4], [2], [4]])

# Calculate predicted prices using the learned model
predicted_prices = (
    intercept
    + slope_engine_size * new_engine_size
    + slope_horsepower * new_horsepower
    + slope_vehicle_age * new_vehicle_age
    + slope_num_doors * new_num_doors
)

print("Predicted prices:")
for i in range(len(new_engine_size)):
    print(
        f"Engine Size: {new_engine_size[i][0]} L, Horsepower: {new_horsepower[i][0]} HP, "
        f"Vehicle Age: {new_vehicle_age[i][0]} years, Number of Doors: {new_num_doors[i][0]}, "
        f"Predicted Price: ${predicted_prices[i][0]:.2f}"
    )

# Print the Regression Equation
print(
    f"Regression equation: Price = {intercept:.2f} + {slope_engine_size:.2f} * Engine Size + "
    f"{slope_horsepower:.2f} * Horsepower + {slope_vehicle_age:.2f} * Vehicle Age + {slope_num_doors:.2f} * Number of Doors"
)
