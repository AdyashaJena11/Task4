# Task4
ask 4: Classification with Logistic Regression
This project is a solution for the AI/ML internship task on binary classification using Logistic Regression. The objective is to build a classifier, evaluate its performance, and understand the core concepts behind it.

Objective
Build a binary classifier for the Breast Cancer Wisconsin Dataset to predict whether a tumor is malignant or benign.
Tools Used:
Python
Scikit-learn: For machine learning models and metrics.
Pandas: For data manipulation and analysis.
Matplotlib & Seaborn: For data visualization.
Steps Taken
The project follows the steps outlined in the task description:
Train/Test Split: The data was split into an 80% training set and a 20% testing set. stratify=y was used to ensure that the proportion of target classes was the same in both the train and test sets.
Standardize Features: The features were standardized using StandardScaler. This process scales the data to have a mean of 0 and a standard deviation of 1. It's a crucial step for logistic regression because the algorithm uses regularization, which is sensitive to the scale of features.
Fit Logistic Regression Model: A LogisticRegression model was trained on the scaled training data.
Evaluate the Model: The model's performance was evaluated on the test set using several key metrics:
Confusion Matrix: To visualize the number of correct and incorrect predictions (True Positives, True Negatives, False Positives, False Negatives).
Classification Report: To calculate precision, recall, and F1-score for each class.
ROC-AUC Score & Curve: To measure the model's ability to distinguish between the two classes across all thresholds. The Area Under the Curve (AUC) provides a single score for this performance.
Threshold Tuning: The effect of changing the decision threshold from the default of 0.5 was demonstrated. This showed the trade-off between precision and recall.
How to Run the Code
Clone this repository.
Make sure you have Python and the required libraries installed:
pip install scikit-learn pandas matplotlib seaborn
Run the Python script:
python logistic_regression_classifier.py
The script will print the results of each step to the console and save three plots as PNG files:
confusion_matrix.png
roc_curve.png
sigmoid_function.png
Results
The trained logistic regression model performed very well on the test set, achieving:
Accuracy: ~97%
ROC-AUC Score: ~0.99
This indicates that the model is highly effective at distinguishing between malignant and benign tumors based on the given features.
