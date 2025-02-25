import os
import pickle
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
import mlp
import matplotlib.pyplot as plt
import numpy as np

FEATURES_FILENAME = 'mpg_features.pkl'
TARGETS_FILENAME = 'mpg_targets.pkl'

if os.path.exists(FEATURES_FILENAME) and os.path.exists(TARGETS_FILENAME):
    with open(FEATURES_FILENAME, 'rb') as f:
        X = pickle.load(f)
        print("Loaded features from " + FEATURES_FILENAME)
    with open(TARGETS_FILENAME, 'rb') as f:
        y = pickle.load(f)
        print("Loaded targets from " + TARGETS_FILENAME)
else:
    auto_mpg = fetch_ucirepo(id=9)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets
    with open(FEATURES_FILENAME, 'wb') as f:
        pickle.dump(X, f)
        print("Loaded features from " + FEATURES_FILENAME)
    with open(TARGETS_FILENAME, 'wb') as f:
        pickle.dump(y, f)
    print("Fetched dataset from ucimlrepo and features to " + FEATURES_FILENAME + " and targets to " + TARGETS_FILENAME)

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (features)
y_mean = y_train.mean(axis=0)  # Mean of each feature
y_std = y_train.std(axis=0)    # Standard deviation of each feature

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

perceptron = mlp.MultilayerPerceptron((mlp.Layer(7, 128, mlp.Relu()),mlp.Layer(128, 256, mlp.Relu()), mlp.Layer(256, 64, mlp.Relu()), mlp.Layer(64, 1, mlp.Linear())))

training_loss, validation_loss = perceptron.train(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), mlp.SquaredError(), learning_rate=0.0001, epochs=40, batch_size=256)

plt.plot(training_loss, color='b', label='Training')
plt.plot(validation_loss, color='r',linestyle='dashed', label="Validation")
plt.title("Loss Curve", size=16)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

pred_y = np.round(perceptron.forward(X_test.to_numpy()) * y_std + y_mean)

table = pd.DataFrame({
    'True MPG': y_test.to_numpy().flatten() * y_std + y_mean,
    'Predicted MPG': pred_y.flatten() 
})
print("\nSample Predictions on Test Data:")
print(table)

print(f"Final Training Loss: {training_loss[-1]:.4f}")
print(f"Final Validation Loss: {validation_loss[-1]:.4f}")