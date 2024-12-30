import numpy as np

# did 0.5 then 0.1 for learning rate
class MLP:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.NI = num_inputs
        self.NH = num_hidden
        self.NO = num_outputs
        
        # Randomly initialize weights
        self.W1 = np.random.uniform(-0.1, 0.1, (self.NI, self.NH))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.NH, self.NO))
        
        # Initialize deltas
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        
        # Activations
        self.Z1 = np.zeros(self.NH)
        self.Z2 = np.zeros(self.NO)
        self.H = np.zeros(self.NH)
        self.O = np.zeros(self.NO)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, I):
        self.Z1 = np.dot(I, self.W1)
        self.H = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.H, self.W2)
        self.O = self.sigmoid(self.Z2)
        return self.O

    def backward(self, I, T):
        # Calculate output error
        output_error = T - self.O
        output_delta = output_error * self.sigmoid_derivative(self.O)

        # Calculate hidden error
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.H)

        # Calculate weight updates
        self.dW2 = np.outer(self.H, output_delta)
        self.dW1 = np.outer(I, hidden_delta)

        # Return mean squared error
        return np.mean(output_error ** 2)
    
    def update_weights(self, learning_rate):
        self.W1 += learning_rate * self.dW1
        self.W2 += learning_rate * self.dW2
        self.dW1.fill(0)
        self.dW2.fill(0)

# -------------------------------------------------------------------------------

# XOR dataset
inputs = np.array([[0, 0], 
                   [0, 1], 
                   [1, 0], 
                   [1, 1]]
                 )

targets = np.array([[0], 
                    [1], 
                    [1], 
                    [0]]
                  )

# Hyperparameters
num_inputs = 2
num_hidden = 16 # did 4, 8 to find different results
num_outputs = 1
learning_rate = 1 # did 0.1, 0.01, 0.5 to find different results
epochs = 10001

# Initialize MLP
mlp = MLP(num_inputs, num_hidden, num_outputs)

# Training
for epoch in range(epochs):
    error = 0
    for x, t in zip(inputs, targets):
        mlp.forward(x)
        error += mlp.backward(x, t)
        mlp.update_weights(learning_rate)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Error: {error:.6f}")

# Testing
print("Testing XOR Function")
for x in inputs:
    prediction = mlp.forward(x)
    print(f"Input: {x}, Predicted: {prediction.round()}, Actual: {prediction}")    

# -------------------------------------------------------------------------------

# Generating the Sin() Dataset
np.random.seed(42)  # For reproducibility

# Generate 500 random vectors, each with 4 components between -1 and 1
inputs = np.random.uniform(-1, 1, (500, 4))
# Calculate the corresponding output as sin(x1 - x2 + x3 - x4)
targets = np.sin(inputs[:, 0] - inputs[:, 1] + inputs[:, 2] - inputs[:, 3]).reshape(-1, 1)

# Split dataset into training (400 examples) and testing (100 examples)
train_inputs, test_inputs = inputs[:400], inputs[400:] # originally 400
train_targets, test_targets = targets[:400], targets[400:] # originally 400

# Hyperparameters
num_inputs = 4
num_hidden = 5000  # At least 5 hidden units # did 5, 25, 50
num_outputs = 1
learning_rate = 0.001 # did 0.1, 0.01, 0.5, 1, 0.001
epochs = 5001

# Initialize MLP
mlp = MLP(num_inputs, num_hidden, num_outputs)

# Training
print("Training on Sin() function")
for epoch in range(epochs):
    error = 0
    for x, t in zip(train_inputs, train_targets):
        mlp.forward(x)
        error += mlp.backward(x, t)
        mlp.update_weights(learning_rate)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Error: {error:.6f}")

# Testing
print("\nTesting Sin() function")
test_error = 0
for x, t in zip(test_inputs, test_targets):
    prediction = mlp.forward(x)
    test_error += np.mean((t - prediction) ** 2)
    print(f"Input: {x}, Predicted: {prediction[0]:.6f}, Actual: {t[0]:.6f}")

# Report Mean Squared Error on Test Set
test_error /= len(test_inputs)
print(f"\nMean Squared Error on Test Set: {test_error:.6f}")

# -------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the dataset from the local file
file_path = "letter_recognition.data"
columns = ["Letter"] + [f"Feature_{i}" for i in range(1, 17)]
data = pd.read_csv(file_path, names=columns)

# Convert target letter to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)  # Updated to avoid the FutureWarning
letters = data["Letter"].values.reshape(-1, 1)
targets = encoder.fit_transform(letters)

# Features
inputs = data.iloc[:, 1:].values

# Train-test split (80% train, 20% test)
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)

# Normalize input features to [0, 1]
train_inputs = train_inputs / 15.0
test_inputs = test_inputs / 15.0

# -------------------------------------------------------------------------------

# Hyperparameters
num_inputs = 16  # 16 features
num_hidden = 50  # Start with 10 hidden units # did 10
num_outputs = 26  # One output per letter
learning_rate = 0.5  # did 0.1
epochs = 1001

# Initialize MLP
mlp = MLP(num_inputs, num_hidden, num_outputs)

# Training
print("Training on Letter Recognition Dataset")
for epoch in range(epochs):
    error = 0
    for x, t in zip(train_inputs, train_targets):
        mlp.forward(x)
        error += mlp.backward(x, t)
        mlp.update_weights(learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error:.6f}")

# Testing
print("\nTesting Letter Recognition Dataset")
correct_predictions = 0
for x, t in zip(test_inputs, test_targets):
    prediction = mlp.forward(x)
    predicted_label = prediction.argmax()  # Select the index with the highest probability
    actual_label = t.argmax()
    if predicted_label == actual_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(test_inputs)
print(f"\nAccuracy on Test Set: {accuracy:.6f}")
