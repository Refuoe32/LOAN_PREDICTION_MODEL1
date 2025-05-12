# Loan Prediction System

 ## Project Overview
 This project develops a machine learning model to predict loan approval outcomes for Motz Financial Services. The model uses applicant data such as gender, marital status, income, and credit history to determine loan eligibility. A key focus is identifying the most important features influencing loan approval. Preprocessed training and testing datasets are saved for inspection.

 ## Dataset
 The dataset (`loan_data_set.csv`) contains applicant details and loan status. Key features include:
 - ApplicantIncome
 - CoapplicantIncome
 - LoanAmount
 - Credit_History
 - Loan_Status (target)

 ## Methodology
 1. **Data Cleaning**: Handled missing values using mode for categorical and median for numerical features; removed outliers with IQR method.
 2. **Preprocessing**: Created Total_Income feature, encoded categorical variables, scaled numerical features.
 3. **Data Splitting**: Randomly split into 80% training and 20% testing sets.
 4. **Model**: Used Logistic regression for its accurancy compared to other models.
 5. **Tuning**: Applied GridSearchCV to optimize hyperparameters.
 6. **Feature Importance**: Analyzed and ranked features to identify those most critical for loan approval.
 7. **Dashboard**: Built a dashboard with interactive visualizations and feature importance insights and data visualisations.

 ## Files
 - `loan_prediction_model.py`: Main script for model training and dashboard
 - `loan_prediction_model.pkl`: Trained XGBoost model
 - `train_data_preprocessed.csv`: Preprocessed 80% training dataset
 - `test_data_preprocessed.csv`: Preprocessed 20% testing dataset
 - `test_predictions.csv`: Test set predictions
 - `model_performance.txt`: Model accuracy and parameters
 - `feature_importance_scores.txt`: Top feature importance scores
 - `model_details.txt`: Model creation details
 - Visualization images in `outputs/`

 ## Usage
 1. Install dependencies: `pip install pandas numpy sklearn xgboost matplotlib seaborn joblib streamlit`
 2. Open the displayed URL in a browser to view the dashboard.
 3. Use the trained model for predictions with `joblib.load('loan_prediction_model.pkl')`.
 4. Inspect preprocessed datasets in `outputs/train_data_preprocessed.csv` and `outputs/test_data_preprocessed.csv`.

 ## Requirements
 - Python 3.8+
 - Libraries: pandas, numpy, sklearn, matplotlib, seaborn, joblib, streamlit

 ## Team
 REFUOE,SELLO,RAMONE,MOSOLOLI,HEQOA]

 
