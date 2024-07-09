## LoanStatus-Predictor-Analyzing-Factors-Influencing-Loan-Repayment

## Problem Statement
The dataset aims to analyze the factors influencing loan repayment behavior to improve credit risk assessment models.

## Objectives
Identify key factors (such as income, employment status) influencing loan default rates.
Build predictive models to forecast loan repayment likelihood.
Provide insights to improve loan approval processes and reduce credit risks.

## Methodology
Data Understanding and Preparation:
Data Loading: Load the dataset using pd.read_csv() and inspect its structure using df.head() to understand the initial format.
Data Information: Use df.info() to understand the data types and missing values (df.isna().sum()).
Handling Missing Values: Fill missing values with the mean using df.fillna(df.mean(), inplace=True).
Data Encoding:
Label Encoding: Use LabelEncoder from sklearn.preprocessing to convert categorical variables into numerical representations suitable for modeling (le.fit_transform() for each categorical column).
Data Splitting:
Train-Test Split: Split the dataset into training and testing sets using train_test_split() from sklearn.model_selection.
Model Building:
Logistic Regression Model: Initialize a logistic regression model using LogisticRegression() from sklearn.linear_model.
Model Training: Fit the logistic regression model on the training data using log_reg.fit(X_train, y_train).
Model Evaluation:
Prediction: Predict loan statuses on the test set using log_reg.predict(X_test).
Accuracy Calculation: Compute the accuracy score of the model using accuracy_score(y_test, y_pred).
Confusion Matrix: Generate and display the confusion matrix to understand the model's performance metrics like true positives, true negatives, false positives, and false negatives (confusion_matrix(y_test, y_pred)).
Classification Report: Generate a detailed classification report showing precision, recall, and F1-score for each class using classification_report(y_test, y_pred).
Visualization:
Confusion Matrix Visualization: Plot a heatmap of the confusion matrix using matplotlib and seaborn to visualize the distribution of true and predicted labels (sns.heatmap()).


## Results
### Model Performance
- **Accuracy:** 89%

### Classification Report
         precision    recall  f1-score   support

      0       0.85      0.92      0.88       201
      1       0.92      0.85      0.88       219

These metrics indicate strong predictive performance of the logistic regression model in identifying loan statuses based on the dataset features.
