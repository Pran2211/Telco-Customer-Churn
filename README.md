Telco Customer Churn Prediction Project 📊
Overview
This project focuses on building a machine learning model to predict customer churn in a telecommunications company. By analyzing various customer attributes and service usage patterns, the model aims to identify customers who are likely to discontinue their services, enabling proactive retention strategies.

Features ✨
Comprehensive Data Preprocessing: Handles missing values, performs data type conversions (e.g., TotalCharges), and standardizes categorical data.

Extensive Exploratory Data Analysis (EDA): Visualizes feature distributions, identifies and addresses class imbalance in the target variable.

Feature Engineering: Transforms and scales numerical features, and encodes categorical features for model readiness.

Class Imbalance Handling: Employs SMOTE (Synthetic Minority Over-sampling Technique) to balance the training dataset, crucial for accurate churn prediction.

Logistic Regression Model: Implements and evaluates a Logistic Regression classifier, including strategic use of class_weight to manage imbalance.

Detailed Model Evaluation: Provides accuracy, confusion matrix, classification report (precision, recall, F1-score), ROC curve, and a precision-recall vs. threshold plot for in-depth performance analysis.

Threshold Tuning: Demonstrates the ability to select an optimal prediction threshold based on business needs (e.g., prioritizing recall for churn detection).

Dataset 📚
The project uses the Telco Customer Dataset.csv file. This dataset contains customer demographic information, service subscriptions, billing details, and their churn status.

Columns: 21

Rows: 7043

Target Variable: Churn (binary: 'No' / 'Yes')

Key Challenge: Significant class imbalance in the Churn variable (approx. 73.5% 'No', 26.5% 'Yes').

Methodology 🛠️
The project follows a structured machine learning workflow:

Data Loading & Initial Cleaning:

Loads Telco Customer Dataset.csv.

Drops customerID.

Converts TotalCharges to numeric, imputing empty strings with the median.

Data Preprocessing & Feature Engineering:

Maps binary categorical features (gender, Partner, Dependents, etc.) to numerical (0/1).

Consolidates categories in service-related columns (e.g., 'No internet service' to 'No').

Applies np.log1p transformation to skewed numerical features (tenure, MonthlyCharges, TotalCharges).

Scales all numerical features using StandardScaler.

Applies One-Hot Encoding to all remaining categorical features.

Data Splitting & Resampling:

Splits the data into an 80% training set and a 20% test set.

Applies SMOTE to the training data to balance the Churn classes.

Model Training:

A LogisticRegression model is trained on the SMOTE-resampled training data.

class_weight={0: 1, 1: 1.5} is used to give more importance to the minority class during training.

Model Evaluation:

The model's performance is assessed on the unseen test set using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC Curve and AUC score

Precision-Recall vs. Threshold plot to identify an optimal operating point.

Results 📊
The Logistic Regression model, trained with SMOTE and class weighting, demonstrated effective churn prediction on the test set.

Accuracy on Final Test Set: Approximately 78%

Key Insights from Classification Report (Churn=Yes class):

Precision: [Insert Precision for 'Yes' from your output, e.g., 0.57] - This indicates that [e.g., 57%] of customers predicted to churn actually churned.

Recall: [Insert Recall for 'Yes' from your output, e.g., 0.64] - This indicates that [e.g., 64%] of actual churners were correctly identified by the model.

F1-Score: [Insert F1-Score for 'Yes' from your output, e.g., 0.60] - This provides a balanced measure of precision and recall.

The Precision-Recall vs. Threshold plot is particularly valuable, showing how the model's performance changes at different probability cutoffs, allowing for a strategic choice based on business objectives (e.g., if identifying as many churners as possible is critical, a lower threshold might be chosen to maximize recall, even if it slightly reduces precision).

How to Run 🚀
Prerequisites
Python 3.x

Jupyter Notebook (recommended for interactive exploration)

Installation
Clone the repository or download the project files.

Navigate to the project directory in your terminal.

Install the required Python libraries:

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

(Note: imbalanced-learn is for SMOTE.)

Running the Code
Ensure Data: Place your Telco Customer Dataset.csv file in the same directory as the script, or update the os.chdir path in the script to point to the correct location of your data.

Execute Script:

Using Jupyter Notebook: Open Telco Customer Churn Project Notebook.ipynb (or Telco Customer Churn Project Script.py if converted to .ipynb) and run all cells sequentially.

As a Python Script:

python "Telco Customer Churn Project Script.py"

Dependencies 📦
numpy

pandas

matplotlib

seaborn

scikit-learn

imbalanced-learn

Future Enhancements 🔮
Explore More Models: Investigate other classification algorithms such as Gradient Boosting (XGBoost, LightGBM), Random Forests, or even simple Neural Networks.

Hyperparameter Tuning: Implement GridSearchCV or RandomizedSearchCV for more systematic hyperparameter optimization across various models.

Advanced Feature Engineering: Create more complex interaction features or derive new features from existing ones (e.g., customer lifetime value segments).

Ensemble Methods: Experiment with ensemble techniques like stacking or blending to combine the strengths of multiple models.

Explainable AI (XAI): Use tools like SHAP or LIME to interpret model predictions and understand the key drivers of churn for individual customers, providing actionable insights for business teams.

Time-Series Analysis: If more granular historical data is available, consider time-series models to capture evolving customer behavior patterns.

Deployment: Develop a simple web application or API to demonstrate the model's real-time churn prediction capabilities.
