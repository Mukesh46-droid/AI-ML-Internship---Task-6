# AI-ML-Internship---Task-6
# K-Nearest Neighbors (KNN) Classification

## Project Overview
This project implements the K-Nearest Neighbors (KNN) algorithm for classification tasks using Python. The Iris dataset is used to demonstrate classification with KNN, including data normalization, training, evaluating with different values of K, and visualizing decision boundaries.

## Tools and Libraries Used
*   Python 3.x
*   Scikit-learn
*   Pandas
*   Matplotlib
*   NumPy

## Features
*   Loads and preprocesses the Iris dataset.
*   Normalizes feature values using `StandardScaler`.
*   Trains a KNN classifier with different values of K (1, 3, 5, 7).
*   Evaluates the model's performance using accuracy scores and confusion matrices.
*   Includes functionality to visualize the decision boundaries for understanding classifier behavior.

## Usage
1.  Clone or download the project files.
2.  Install the required Python libraries using pip:
    ```
    pip install scikit-learn pandas matplotlib numpy
    ```
3.  Run the Python script to train and evaluate the KNN classifier. The script will output the accuracy and confusion matrix for each value of K.

## How KNN Works
KNN is a lazy, instance-based learning algorithm. It does not learn a model from the training data but instead stores the entire dataset. To make a prediction, it identifies the 'K' most similar instances (neighbors) from the training data and returns the majority class among them. The similarity is typically calculated using a distance metric like Euclidean distance.

## Results
The performance of the KNN classifier was evaluated on the Iris dataset.
*   Accuracy varies with the choice of K, with an optimal value balancing bias and variance.
*   For the subset of the Iris dataset used (two features), the accuracy stabilized at **80.0%** for K values of 3, 5, and 7.
*   Decision boundary plots help visualize how the KNN classifier separates the different classes based on the value of K.


