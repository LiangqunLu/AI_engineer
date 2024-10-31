# Machine Learning Fundamentals

## 1. Mathematical Foundations

### Linear Algebra
- **Vectors and Matrices**
  - Vector operations
  - Matrix multiplication
  - Transpose, inverse, determinant
  - Eigenvalues and eigenvectors
  - Singular Value Decomposition (SVD)

### Calculus
- **Derivatives**
  - Chain rule
  - Partial derivatives
  - Gradients
  - Jacobian and Hessian matrices
- **Optimization**
  - Gradient descent variants
  - Convex optimization
  - Lagrange multipliers

### Probability & Statistics
- **Probability Basics**
  - Probability distributions
  - Conditional probability
  - Bayes' theorem
  - Maximum likelihood estimation
- **Statistical Measures**
  - Mean, median, mode
  - Variance, standard deviation
  - Correlation, covariance
  - Hypothesis testing
  - Confidence intervals

## 2. Core ML Concepts

### Model Training
- **Loss Functions**
  - MSE (Mean Squared Error)
  - Cross-entropy loss
  - Hinge loss
  - Huber loss

- **Optimization Algorithms**
  - Gradient Descent
  - Stochastic Gradient Descent
  - Mini-batch SGD
  - Adam, RMSprop, AdaGrad
  - Learning rate scheduling

- **Regularization**
  - L1 regularization (Lasso)
  - L2 regularization (Ridge)
  - Elastic Net
  - Dropout
  - Early stopping

### Model Evaluation
- **Metrics**
  - Classification: accuracy, precision, recall, F1-score, ROC-AUC
  - Regression: MSE, MAE, R-squared, RMSE
  - Ranking: NDCG, MAP, MRR

- **Validation Techniques**
  - Train-test split
  - Cross-validation
  - Stratified sampling
  - Holdout validation

### Bias-Variance Trade-off
- Underfitting vs Overfitting
- Model complexity
- Bias-variance decomposition
- Generalization error

## 3. Classical ML Algorithms

### Supervised Learning
- **Linear Models**
  - Linear Regression
  - Logistic Regression
  - Support Vector Machines (SVM)

- **Tree-based Models**
  - Decision Trees
  - Random Forests
  - Gradient Boosting (XGBoost, LightGBM)

- **Other Algorithms**
  - k-Nearest Neighbors (kNN)
  - Naive Bayes
  - Ensemble methods

### Unsupervised Learning
- **Clustering**
  - K-means
  - DBSCAN
  - Hierarchical clustering
  - Gaussian Mixture Models

- **Dimensionality Reduction**
  - PCA (Principal Component Analysis)
  - t-SNE
  - UMAP
  - Autoencoders

## 4. Deep Learning Fundamentals

### Neural Networks
- **Architecture Components**
  - Neurons and layers
  - Activation functions
  - Forward/backward propagation
  - Weight initialization

- **Common Architectures**
  - Feedforward Neural Networks
  - CNNs (Convolutional Neural Networks)
  - RNNs (Recurrent Neural Networks)
  - Transformers

### Deep Learning Concepts
- **Training Techniques**
  - Batch normalization
  - Layer normalization
  - Transfer learning
  - Fine-tuning

- **Advanced Topics**
  - Attention mechanisms
  - Self-supervised learning
  - Few-shot learning
  - Contrastive learning

## 5. Feature Engineering

### Feature Processing
- **Scaling**
  - Min-max scaling
  - Standard scaling
  - Robust scaling
  - Normalization

- **Encoding**
  - One-hot encoding
  - Label encoding
  - Target encoding
  - Feature hashing

### Feature Selection
- **Methods**
  - Filter methods
  - Wrapper methods
  - Embedded methods
  - Feature importance

## 6. Practical Considerations

### Data Processing
- **Data Cleaning**
  - Handling missing values
  - Outlier detection
  - Data imputation
  - Data validation

- **Data Imbalance**
  - Oversampling (SMOTE)
  - Undersampling
  - Class weights
  - Ensemble approaches

### Model Selection
- **Selection Criteria**
  - Cross-validation scores
  - Model complexity
  - Inference time
  - Resource constraints

- **Hyperparameter Tuning**
  - Grid search
  - Random search
  - Bayesian optimization
  - Population-based training

## 7. Common Pitfalls and Best Practices

### Common Issues
- Data leakage
- Selection bias
- Correlation vs causation
- Class imbalance
- Curse of dimensionality

### Best Practices
- Start simple, then complexify
- Always have a baseline
- Cross-validate properly
- Monitor model performance
- Document assumptions
- Version control for data and models

## 8. Essential Tools & Libraries

### Python Ecosystem
- NumPy for numerical computing
- Pandas for data manipulation
- Scikit-learn for classical ML
- PyTorch/TensorFlow for deep learning
- Matplotlib/Seaborn for visualization

### Development Tools
- Jupyter Notebooks
- Version control (Git)
- Experiment tracking
- Model versioning
- Deployment tools

# Top ML Fundamental Interview Questions

## 1. Core ML Concepts

1. **What is the difference between supervised and unsupervised learning?**
   - Supervised: Labeled data, predicts output (classification/regression)
   - Unsupervised: Unlabeled data, finds patterns (clustering/dimensionality reduction)
   - Examples for each
   - Common algorithms

2. **Explain bias-variance tradeoff**
   - Bias: Model's assumptions about data
   - Variance: Model's sensitivity to fluctuations
   - Trade-off relationship
   - How to find optimal balance
   - Examples of high bias/variance

3. **What's the difference between L1 and L2 regularization?**
   - L1 (Lasso): Sum of absolute weights, promotes sparsity
   - L2 (Ridge): Sum of squared weights, prevents large weights
   - When to use each
   - Effect on model parameters
   - Mathematical formulation

4. **Explain gradient descent and its variants**
   - Basic gradient descent
   - Stochastic gradient descent
   - Mini-batch gradient descent
   - Advantages/disadvantages
   - Common challenges (local minima, learning rate)

5. **What are evaluation metrics and when to use them?**
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
   - Regression: MSE, MAE, R-squared
   - When to use each
   - Trade-offs between metrics

## 2. Essential Statistical Concepts

1. **Explain the Central Limit Theorem**
   - Definition
   - Importance in ML
   - Applications
   - Assumptions
   - Examples

2. **What is p-value?**
   - Definition
   - Interpretation
   - Common misconceptions
   - Use in hypothesis testing
   - Limitations

3. **Explain correlation vs causation**
   - Definitions
   - Differences
   - Examples
   - Common pitfalls
   - How to determine causality

4. **What is selection bias?**
   - Types of selection bias
   - How it affects models
   - How to detect it
   - How to mitigate it
   - Real-world examples

5. **Explain Bayes' Theorem and its applications**
   - Formula
   - Intuition
   - Applications in ML
   - Examples
   - Naive Bayes classifier

## 3. Algorithm Understanding

1. **How does logistic regression work?**
   - Mathematical formulation
   - Assumptions
   - Use cases
   - Advantages/disadvantages
   - Difference from linear regression

2. **Explain decision trees**
   - How they work
   - Splitting criteria
   - Advantages/disadvantages
   - Preventing overfitting
   - Real-world applications

3. **How do neural networks learn?**
   - Architecture
   - Backpropagation
   - Activation functions
   - Gradient descent
   - Common challenges

4. **What is the difference between bagging and boosting?**
   - Bagging (Random Forests)
   - Boosting (AdaBoost, XGBoost)
   - When to use each
   - Advantages/disadvantages
   - Examples

5. **Explain SVM and kernel trick**
   - Linear SVM
   - Kernel trick
   - Common kernels
   - When to use
   - Advantages/disadvantages

## 4. Feature Engineering

1. **How do you handle missing data?**
   - Types of missing data
   - Imputation strategies
   - When to remove data
   - Impact on model
   - Best practices

2. **Explain feature scaling methods**
   - Standardization
   - Normalization
   - When to use each
   - Impact on models
   - Implementation considerations

3. **How do you handle categorical variables?**
   - One-hot encoding
   - Label encoding
   - Target encoding
   - When to use each
   - Handling high cardinality

4. **What is dimensionality reduction?**
   - PCA
   - t-SNE
   - When to use
   - Advantages/disadvantages
   - Real-world applications

5. **How do you handle imbalanced datasets?**
   - Oversampling
   - Undersampling
   - SMOTE
   - Class weights
   - Evaluation considerations

## 5. Model Validation

1. **Explain cross-validation**
   - K-fold cross-validation
   - Stratified k-fold
   - Leave-one-out
   - When to use each
   - Implementation considerations

2. **What is overfitting and how to prevent it?**
   - Signs of overfitting
   - Prevention techniques
   - Regularization
   - Early stopping
   - Cross-validation

3. **How do you choose model hyperparameters?**
   - Grid search
   - Random search
   - Bayesian optimization
   - Cross-validation
   - Trade-offs

4. **Explain ROC curve and AUC**
   - What they measure
   - How to interpret
   - When to use
   - Limitations
   - Alternatives (PR curve)

5. **What is the difference between parameters and hyperparameters?**
   - Definitions
   - Examples
   - How they're chosen
   - Optimization methods
   - Impact on model performance