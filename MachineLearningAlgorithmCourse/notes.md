# Machine Learning
## Data Preprocessing includes:
  -   Importing the lib
  -   Importing the dataset
  -   take care of missing data
  -   Encode categorical data
      -   Encode independent variables
      -   Encode dependent variables
  -   Split train and test data
  -   feature scaling
### How to deal with missing data?
- Delete the row and ignore
- Replace the missing values with average of all the salaries.
### What to do with categorical data?
- Encode it
  - Dependant variable
  - Independent variable
- One Hot Encoding : for example for countries. when there is no ordering.
- Label Encoding: No|Yes for Y can be encoded to 0,1
### Feature scaling
- To have all teh values of features in same range
- Should not be done at features which were encoded
- Only for numerical values
- Two types
  - Standardization (-2 to +2)
  - Normalization (0 to 1)
## Regression
 - To predict the real value like salary
 - [GeeksForGeeks Article](https://www.geeksforgeeks.org/types-of-regression-techniques/)
### Simple Linear Regression
### Multiple Linear Regression
- No need to apply feature scaling in multiple linear regression
## Building a Model
- All IN
- Backward Illimination (fastest)
- Forward Selection
- Bidirectional Elimination
- Score Comparison
### Polynomial Regression
### Support Vector for Regression (SVR)
### Decision Tree Regression
### Random Forest Regression

## SVR - Support Vector Regression
- A tube instead of a linear line

## Classification

## Logistic Regression
- Used to calculate Category like yes or no
- To get best curve calculate curve with best likelihood value.

## K-Nearest Neighbor
- Calculate the distance of neighbor's upto k for ex 5 and put the new point in that category where max points are there.

## SVM
- Lines between extreme one type of point or classification like apples (Negative hyperplane) and first another type of point like oranges (positive hyperplane) points.
- Used to classify apples or oranges.

## Decision tree classification
- Yes/No ladder
- Old method
- reborn with upgrades
  - random forest
  - gradient boosting

## Random forest classification
- ensemble learning : multiple algo to create one algo

<hr/>

## Clustering

### K-Means Clustering
- Not limited to 2 dimensions. Can be scaled to many dimensions.
- Calculate center of mass to come to conclusion for getting centroids
- The elbow method : To find how many clusters are available.
- K-Means++ to get consistent results.
- We create dependent variable

### Hierarchial Clustering
- Same as K means but with different process
- Two types
  - Agglomerative
  - Divisive
- Dendrograms - to get optimal number of clusters

### Association Rule Learning
- Customers who bought also bought
- Two models
  - Apriori: SCL: Support confidence lift
  - Eclat: ARL

### Reinforcement Learning
- Multi-Armed Bandit Problem
  - Upper confidence bound (UCB):
    - Is deterministic
  - Thompson Sampling:
    - is probabilistic

<hr/>

### Natural Language Processing (NLP)
- Types (Some of them)
  - Nlp
  - Deep Learning
  - Deep NLP
  - Seq2Seq
- Bag of words model
  - Creating an array of words

<hr/>

## Deep Learning

### Artificial Neural Networks

### Gradient Descent
### Stochastic Gradient Descent
### Backpropagation

### Convolutional Neural Networks
- Image recognition
- Convolutional layer
- used to detect emotions
- https://adamharley.com/nn_vis/cnn/2d.html

## Dimensionality Reduction
- PCA
- LDA
- Kernel PCA

### Model Selection
- K-Fold Cross Validation
- Grid Search
- XGBoost


<hr/>

# General Steps
- Import Libraries
- Split dataset into training and testing set
- Feature scaling
- Training
- Predicting single result
- Predicting test results


# Questions?
- When to use SVR?
- What is feature scaling? When to use it?
  - Linear reg should not use feature scaling
  - Implicit equation and implicit relation between dep variable y. we should apply feature scaling
  - When to use decision tree regression vs Linear regression?
- What is confusion matrix?
- What is Naive Based?
- Accuracy paradox vs cumulative accuracy paradox vs Receiver Operating Characteristic?
- Supervised vs Unsupervised trainings.
