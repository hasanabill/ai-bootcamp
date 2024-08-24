import numpy as np
import matplotlib as plt

np.random.seed(0)
sqft_train = np.random.randint(1000, 3000, 1000)

# print(sqft_train)
# print(sqft_train.shape)

price_train = 50 * sqft_train + np.random.normal(0, 5000, 1000)

# print(price_train)
for s, p in zip(sqft_train[0:1000], price_train[0:1000]):
    print(s, p)
