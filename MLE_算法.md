MLE 面试准备
Linear regression; Logistic regression; decision tree; random forest; NaiveBayes; SVM; KNN; NN; and K means clustering


Reference: https://chat.openai.com/g/g-Ku5sL9Qgm-musk-mentor/c/17806dad-7703-44ee-9b32-76d397d743f2






# **K-Means Algorithm**

### **Theory:**
- **Initialization**:
  - Number of clusters (k)
  - Max iterations
  - Tolerance (threshold for stopping criterion)
- **Centroid Initialization**: Centroids can be initialized randomly or using k-means++ for better initial guesses.
- **Cluster Assignment**: Assign each data point to the nearest centroid using a distance metric like Euclidean distance.
- **Centroid Recalculation**: Update the centroids by calculating the mean of the points in each cluster.
- **Convergence Check**: Check if the centroids have changed by less than the tolerance or if the maximum number of iterations is reached.
- **Repetition**: Repeat the process until convergence.
- **Prediction**: For new data points, assign them to the nearest centroid.

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=0.0001):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = {}

    def fit(self, data):
        # Initialize centroids
        self.centroids = {i: data[i] for i in range(self.k)}

        for i in range(self.max_iters):
            self.classes = {i: [] for i in range(self.k)}

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            # Recalculate centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = previous[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        return distances.index(min(distances))

if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11]
    ])

    kmeans = KMeans(k=2)
    kmeans.fit(X)

    for point in X:
        print(f"Point {point} is in cluster {kmeans.predict(point)}")
```        


# **K-Nearest Neighbors (KNN)**

### **Theory:**
- **Initialization**:
  - Number of nearest neighbors (k)
- **Fit**:
  - No training needed (lazy learning algorithm).
- **Prediction**:
  1. **Distance Calculation**: Compute distances between the test point and all training points.
  2. **Find Nearest Neighbors**: Identify the k-nearest points.
  3. **Majority Vote**: Use the labels of the nearest neighbors to determine the class.


```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    X_train = np.array([
        [1, 2], [2, 3], [3, 4],
        [6, 6], [7, 7], [8, 9]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [5, 5], [2, 2], [8, 8]
    ])

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
```



# **Neural Network (NN)**

### **Theory:**
- **Initialization**:
  - Weights, biases, and define network architecture (number of layers, neurons per layer)
- **Activation Function**: Common functions include Sigmoid, ReLU, Tanh, Softmax.
- **Training (Fit method)**:
  - **Forward Pass**: Data flows through the network to produce predictions.
  - **Backward Pass (Backpropagation)**: Gradients are computed and weights are updated using gradient descent.
- **Prediction**: Use the trained network to make predictions on new data.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
        self.biases = [np.random.randn(layers[i], 1) for i in range(1, len(layers))]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.sigmoid(z))
        return activations, zs

    def backward(self, X, y, activations, zs):
        deltas = [activations[-1] - y]
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.sigmoid_derivative(activations[-l])
            delta = np.dot(self.weights[-l+1].T, deltas[-1]) * sp
            deltas.append(delta)
        deltas.reverse()

        gradients_w = [np.dot(delta, activations[i].T) for i, delta in enumerate(deltas)]
        return gradients_w

    def update_params(self, gradients_w):
        self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, gradients_w)]
    
    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            activations, zs = self.forward(X)
            gradients_w = self.backward(X, y, activations, zs)
            self.update_params(gradients_w)

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

if __name__ == "__main__":
    # Example neural network for binary classification (2 input features, 2 neurons in hidden layer, 1 output)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork([2, 2, 1], learning_rate=0.1)
    nn.fit(X.T, y.T, epochs=5000)

    for point in X:
        prediction = nn.predict(point.T)
        print(f"Input: {point} -> Prediction: {prediction}")
```

# **Linear Regression**

### **Theory:**
- **Initialization**: Define the coefficients (weights) and intercept (bias).
- **Fit by Normal Equation**: Solve the equation \( W = (X^T X)^{-1} X^T y \).
- **Fit by Gradient Descent**: Update weights iteratively to minimize the cost function (mean squared error).
- **Prediction**: \( \hat{y} = XW + b \).
- **Evaluation**: Use metrics like Total Sum of Squares (SS\_tot), Residual Sum of Squares (SS\_res), and \( R^2 \).

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        for i in range(self.iterations):
            self.update_weights(X, y)

    def update_weights(self, X, y):
        y_pred = self.predict(X)
        dW = -(2 * (X.T).dot(y - y_pred)) / self.m
        db = -2 * np.sum(y - y_pred) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.W) + self.b

if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    regressor = LinearRegression(learning_rate=0.01, iterations=1000)
    regressor.fit(X, y)
    predictions = regressor.predict(X)
    print("Predictions:", predictions)        
```

# **Logistic Regression**

### **Theory:**
- **Initialization for Gradient Descent**: Weights, bias, learning rate, and number of iterations.
- **Sigmoid Function**: \( \sigma(z) = \frac{1}{1 + e^{-z}} \).
- **Fit by Gradient Descent**: Minimize the logistic loss (cross-entropy).
- **Prediction**: Use sigmoid function to estimate probabilities and classify based on a threshold.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        for i in range(self.iterations):
            self.update_weights(X, y)

    def update_weights(self, X, y):
        y_pred = self.sigmoid(self.predict_raw(X))
        dW = (X.T).dot(y_pred - y) / self.m
        db = np.sum(y_pred - y) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict_raw(self, X):
        return X.dot(self.W) + self.b

    def predict(self, X):
        return np.round(self.sigmoid(self.predict_raw(X)))

if __name__ == "__main__":
    X = np.array([[0, 1], [1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
```

# **Decision Tree**

### **Theory:**
- **Initialization**: Max depth and other tree parameters.
- **Node Structure**: Internal nodes represent conditions, and leaf nodes represent classes.
- **Best Split Calculation**: Find the split that maximizes information gain (or minimizes impurity).
- **Prediction**: Traverse the tree for each data point and predict the class.

```python
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'depth': depth, 'predicted_class': predicted_class}

        if depth < self.max_depth:
            idx, threshold = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = threshold
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        pass

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if 'left' not in node:
            return node['predicted_class']
        if x[node['feature_index']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    predictions = tree.predict(X)
    print("Predictions:", predictions)

```

# **Random Forest**

### **Theory:**
- **Initialization**: Number of trees, max depth, and other tree hyperparameters.
- **Bootstrap Samples**: Sample data with replacement to create different datasets for each tree.
- **Subset of Features**: At each node, randomly select a subset of features for splitting.
- **Prediction**: Aggregate predictions from all trees.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [np.bincount(tree_pred.astype('int')).argmax() for tree_pred in tree_preds.T]

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])

    forest = RandomForest(n_trees=3, max_depth=3)
    forest.fit(X, y)
    predictions = forest.predict(X)
    print("Predictions:", predictions)
```

# **Naive Bayes**

### **Theory:**
- **Initialization**: Calculate prior probabilities and conditional probabilities for each class.
- **Prediction**: Use Bayes' theorem to compute the posterior probabilities and predict the class with the highest posterior.

```python
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

if __name__ == "__main__":
    X = np.array([[1, 20], [2, 21], [3, 22], [4, 23], [5, 24], [6, 25]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = NaiveBayes()
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
```

# **Support Vector Machine (SVM)**

### **Theory:**
- **Initialization**: Define the kernel, regularization parameter \( C \), and learning rate for optimization.
- **Fit**: Solve the optimization problem to maximize the margin between support vectors.
- **Prediction**: Use the decision boundary to classify new data points.

```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
    y = np.array([0, 0, 0, 1, 1, 1])

    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)
    predictions = svm.predict(X)
    print("Predictions:", predictions)        
```
