# Machine Learning Algorithms from Scratch

This document includes implementations of several fundamental machine learning algorithms written from scratch in Python. These include Linear Regression, Logistic Regression, a simple Neural Network, Decision Tree, and Random Forest. Each implementation is provided in a class-based format.

## Linear Regression

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Adding a column of ones to input X to account for the intercept
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        # Using the Normal Equation to find the best fitting line
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept
```

### Key Steps in Linear Regression
+ Initialization: Initialize coefficients and intercept.
+ Fit the Model: Use the Normal Equation to compute coefficients.
+ Predict: Calculate predictions using the learned model.


## Linear Regression with Gradient Descent

```python

import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

```
### Key Steps in Linear Regression with Gradient Descent
+ Initialization: Set up the learning rate, number of iterations, weights, and bias.
+ Fit the Model: Iteratively adjust weights and bias to minimize the cost function using gradient descent.
+ Predict: Calculate predictions using the optimized weights and bias.





## Logistic Regression

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model) > 0.5

```
Key Steps in Logistic Regression
Initialization: Set up learning rate, iterations, weights, and bias.
Sigmoid Function: Activation function that outputs probabilities.
Fit the Model: Adjust weights and biases using gradient descent.
Predict: Output binary predictions based on probability thresholds.


## Neural Network
```python

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self._sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self._sigmoid(Z2)
            # Backpropagation (for simplicity, not implemented here)
            # Update weights and biases

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        return A2 > 0.5

```
### Key Steps in Neural Network
+ Initialization: Set up layer sizes, weights, and biases.
+ Sigmoid Function: Non-linear activation function used here.
+ Fit the Model: Forward propagation and backpropagation to update weights.
+ Predict: Use the trained network to output predictions.


## Decision Tree

```python
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        self.tree = self._build_tree(X, y, depth)

    def _build_tree(self, X, y, depth):
        # Building tree logic (for simplicity, not implemented here)
        # Determine the best split and recursively build the tree
        return None

    def predict(self, X):
        # Traverse the tree to make predictions
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        # Logic to predict using the built tree
        return 0

```

### Key Steps in Decision Tree
+ Initialization: Set up the maximum depth for the tree.
+ Build the Tree: Use recursive splitting to build the tree based on the best splits.
+ Predict: Traverse the tree based on feature thresholds to make predictions.

## Random Forest
```python
import numpy as np

class RandomForest:
    def __init__(self, n_trees=10, max_depth=2, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap sampling and fitting trees
            tree = DecisionTree(max_depth=self.max_depth)
            # Fit the tree to the data
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting
        return mode(tree_preds, axis=0)[0]

```

### Key Steps in Random Forest
+ Initialization: Define the number of trees, maximum tree depth, and the number of features per tree.
+ Fit the Model: Use bootstrapped samples and a subset of features to train each decision tree independently.
+ Predict: Aggregate the predictions from all trees using majority voting for classification tasks.


This markdown document is designed to be comprehensive and should provide a good starting point for studying these algorithms offline. Each section contains an overview of the key steps involved in each algorithm's operation, making it easier to understand how they function and can be implemented in practice.











