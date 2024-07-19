# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
The purpose of this analysis was to develop a machine learning model that can predict the loan status of a borrower. We wanted to determine whether a given loan would be a healthy loan or a high risk loan. The financial information in the data set included loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, and total debt. In our analysis we needed to predict loan_status and determine if the loan was healthy or high-risk. 

The stages of the machine learning process included data preparation, train-test split, model training, model prediction, and evaluation. 

Data Preparation: 
Data set was loaded into a Pandas DataFrame and viewed the first few rows. Then, deparated the data into the features and target variables. 

Train-Test Split: 
Data was split into training and testing sets in train_test_split. 

Model Training: 
Used the LogisticRegression model with random state of 1 to be reproducable. Then, trained the model with the training data (x_train and y_train)

Model prediction: Used X_test

Evaluation: 
Evaluated model's performance using the confusion matrix. 

Methods used: 
Logistic Regression - This was appropriate for binary classification problems and gives probabilities for each class. 

Confusion Matrix and CLassification Report: 
Used the evaluation metrics to assess the performance of the model. Confusion matrix provided the counts of true positives, negatives, false positives, and negatives. 
## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.
* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
Accuracy Scores:
Accuracy: 0.9917
This states that the model predicts the loan status (healthy or high-risk) 99.17% of the time on the testing data

Precision for class 0 (healthy loan): 0.99
This means that out of all loans predicted as healthy, 99% are actually healthy.

Recall for class 0 (healthy loan): 1.00
This indicates that the model successfully identifies all actual healthy loans.

F1-score for class 0 (healthy loan): 0.99
This is the harmonic mean of precision and recall, providing a single metric that balances both.

Precision for class 1 (high-risk loan): 0.85
This means that out of all loans predicted as high-risk, 85% are actually high-risk.

Recall for class 1 (high-risk loan): 0.38
This indicates that the model successfully identifies 38% of actual high-risk loans.

F1-score for class 1 (high-risk loan): 0.52
This is the harmonic mean of precision and recall, providing a single metric that balances both.



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
The logistic regression model showed high accuracy, particularly in predicting healthy loans. However, it also showed reasonable performance in predicting high-risk loans. The analysis helped us understand the financial characteristics that influence loan status and build a predictive model that can assist financial institutions in making better lending decisions. 
* Which one seems to perform best? How do you know it performs best? 
The logistic regression model performs exceptionally well in predicting loan status. We know this because The model’s precision and recall scores for healthy loans are precise, indicating a balanced performance. The confusion matrix shows that the model correctly identifies nearly all healthy loans and a significant portion of high-risk loans, with only a few misclassifications. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? ) 
Yes, performance depends heavily on the problem we are trying to solve, especially in terms of which class is more critical to predict accurately. In this case, it depends on the financial context and the potential consequences of misclassifications.
If you do not recommend any of the models, please justify your reasoning.
By tailoring the model’s performance to the specific needs of the problem, the financial institution can achieve a better balance between accuracy, precision, and recall, ultimately leading to more effective risk management and customer satisfaction. All in all, it depends on the context of the problem that is to be solved. 