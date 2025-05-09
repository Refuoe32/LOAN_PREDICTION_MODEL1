import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import streamlit as st
import uuid

# Setting up environment for reproducibility
np.random.seed(42)
sns.set_style('darkgrid')

# Creating output directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')


# Apply custom CSS for styling the dashboard
def apply_custom_css():
    with open('loan_dashboard_styles.css', 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Task 1: Data Gathering and Cleaning
# Loading the dataset
def load_and_clean_data():
    df = pd.read_csv('loan_data_set.csv')

    # Print initial missing values information
    print("Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # a. Handling Missing Values
    # Categorical: fill with mode
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    # Numerical: fill with median
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

    # Print missing values after cleaning
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    # b. Handling Outliers
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Applying outlier removal
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numerical_cols:
        df = remove_outliers(df, col)

    print(f"\nDataset Shape After Outlier Removal: {df.shape}")

    # c. Descriptive Analysis
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Generate basic visualizations
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Loan_Status', data=df)
    plt.title('Loan Status Distribution')
    plt.savefig('outputs/loan_status_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(df['ApplicantIncome'], kde=True)
    plt.title('Applicant Income Distribution')
    plt.savefig('outputs/applicant_income_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)
    plt.title('Loan Amount vs. Loan Status')
    plt.savefig('outputs/loan_amount_vs_status.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
    plt.title('Credit History vs. Loan Status')
    plt.savefig('outputs/credit_history_vs_status.png')
    plt.close()

    return df


# Task 2: Data Preprocessing
def preprocess_data(df):
    # Make a copy to preserve original data
    df_processed = df.copy()

    # a. Feature Extraction
    df_processed['Total_Income'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
    df_processed['Income_Ratio'] = df_processed['ApplicantIncome'] / df_processed['Total_Income'].replace(0, 1)
    df_processed['Loan_Term_Ratio'] = df_processed['LoanAmount'] / df_processed['Loan_Amount_Term'].replace(0, 1)

    # Save a copy of the unencoded data with new features (before encoding)
    if 'Loan_ID' in df_processed.columns:
        df_with_features = df_processed.copy()
        df_with_features.to_csv('outputs/data_with_features_pre_encoding.csv', index=False)
        print("\nData with added features before encoding saved to 'outputs/data_with_features_pre_encoding.csv'")

    # b. Feature Encoding
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

    # Create encoders dictionary to store for future use
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df_processed[col] = encoders[col].fit_transform(df_processed[col])

    # Save encoders for future use
    joblib.dump(encoders, 'outputs/label_encoders.pkl')
    print("\nLabel encoders saved to 'outputs/label_encoders.pkl'")

    # Dropping Loan_ID if present
    if 'Loan_ID' in df_processed.columns:
        df_processed = df_processed.drop('Loan_ID', axis=1)

    # Save a copy of the data after encoding but before scaling
    df_processed.to_csv('outputs/data_encoded_pre_scaling.csv', index=False)
    print("\nEncoded data before scaling saved to 'outputs/data_encoded_pre_scaling.csv'")

    # Scaling numerical features
    scaler = StandardScaler()
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income',
                      'Income_Ratio', 'Loan_Term_Ratio']

    # Fit scaler and transform
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

    # Save scaler for future use
    joblib.dump(scaler, 'outputs/numerical_scaler.pkl')
    print("\nNumerical scaler saved to 'outputs/numerical_scaler.pkl'")

    # Save the fully preprocessed dataset
    df_processed.to_csv('outputs/full_preprocessed_data.csv', index=False)
    print("\nFully preprocessed data saved to 'outputs/full_preprocessed_data.csv'")

    return df_processed, scaler, numerical_cols


# Train model and save all necessary files
def train_and_save_model():
    df = load_and_clean_data()
    df_processed, scaler, numerical_cols = preprocess_data(df)

    # Splitting features and target
    X = df_processed.drop('Loan_Status', axis=1)
    y = df_processed['Loan_Status']

    # Randomly splitting data - 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining Set Shape: {X_train.shape}")
    print(f"Testing Set Shape: {X_test.shape}")

    # Save preprocessed training and testing datasets
    train_data = X_train.copy()
    train_data['Loan_Status'] = y_train
    train_data.to_csv('outputs/train_data_preprocessed.csv', index=False)
    print("\nPreprocessed training data saved to 'outputs/train_data_preprocessed.csv'")

    test_data = X_test.copy()
    test_data['Loan_Status'] = y_test
    test_data.to_csv('outputs/test_data_preprocessed.csv', index=False)
    print("Preprocessed testing data saved to 'outputs/test_data_preprocessed.csv'")

    # Using XGBoost
    model = XGBClassifier(random_state=42, eval_metric='logloss')

    # Basic model training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nInitial Model Accuracy: {initial_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    # Model Tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3, 0.5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"\nTuned Model Accuracy: {tuned_accuracy:.4f}")
    print("\nBest Parameters:", grid_search.best_params_)
    print("\nTuned Classification Report:")
    print(classification_report(y_test, y_pred_best))

    # Feature Importance Analysis
    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = feature_importance.nlargest(5).to_dict()
    print("\nTop 5 Most Important Features for Loan Approval:")
    for feature, score in top_features.items():
        print(f"{feature}: {score:.4f}")

    # Save feature importance to CSV
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Importance': feature_importance.values
    }).sort_values('Importance', ascending=False)
    feature_importance_df.to_csv('outputs/feature_importance.csv', index=False)
    print("\nFeature importance saved to 'outputs/feature_importance.csv'")

    # Visualization: Feature Importance
    plt.figure(figsize=(10, 6))
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importance for Loan Approval')
    plt.savefig('outputs/feature_importance.png')
    plt.close()

    # Save predictions
    test_predictions = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_best
    })
    test_predictions.to_csv('outputs/test_predictions.csv', index=False)

    # Save performance metrics
    with open('outputs/model_performance.txt', 'w') as f:
        f.write(f"Initial Accuracy: {initial_accuracy:.4f}\n")
        f.write(f"Tuned Accuracy: {tuned_accuracy:.4f}\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")

    # Save feature importance
    with open('outputs/feature_importance_scores.txt', 'w') as f:
        f.write("Top 5 Most Important Features for Loan Approval:\n")
        for feature, score in top_features.items():
            f.write(f"{feature}: {score:.4f}\n")

    # Save model
    joblib.dump(best_model, 'outputs/loan_prediction_model.pkl')

    # Generate README file
    readme_content = """# Loan Prediction System

 ## Project Overview
 This project develops a machine learning model to predict loan approval outcomes for Motz Financial Services. The model uses applicant data such as gender, marital status, income, and credit history to determine loan eligibility. A key focus is identifying the most important features influencing loan approval. Preprocessed datasets are saved at each step of the processing pipeline.

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
 4. **Model**: Used XGBoost for its robustness and feature importance capabilities.
 5. **Tuning**: Applied GridSearchCV to optimize hyperparameters.
 6. **Feature Importance**: Analyzed and ranked features to identify those most critical for loan approval.
 7. **Dashboard**: Built a Streamlit-based dashboard with interactive visualizations and feature importance insights.

 ## Files
 - `loan_prediction_model.py`: Main script for model training and dashboard
 - `loan_dashboard_styles.css`: CSS styles for the Streamlit dashboard
 - `loan_prediction_model.pkl`: Trained XGBoost model
 - `full_preprocessed_data.csv`: Complete preprocessed dataset
 - `data_with_features_pre_encoding.csv`: Dataset with created features before encoding
 - `data_encoded_pre_scaling.csv`: Dataset after encoding but before scaling
 - `train_data_preprocessed.csv`: Preprocessed 80% training dataset
 - `test_data_preprocessed.csv`: Preprocessed 20% testing dataset
 - `test_predictions.csv`: Test set predictions
 - `model_performance.txt`: Model accuracy and parameters
 - `feature_importance_scores.txt`: Top feature importance scores
 - `feature_importance.csv`: Complete feature importance rankings
 - `label_encoders.pkl`: Saved label encoders for categorical variables
 - `numerical_scaler.pkl`: Saved scaler for numerical features
 - `model_details.txt`: Model creation details
 - Visualization images in `outputs/`

 ## Usage
 1. Install dependencies: `pip install pandas numpy sklearn xgboost matplotlib seaborn joblib streamlit`
 2. Run the script: `streamlit run loan_prediction_model.py`
 3. Open the displayed URL in a browser to view the dashboard.
 4. Use the trained model for predictions with `joblib.load('loan_prediction_model.pkl')`.
 5. Inspect preprocessed datasets in the outputs directory.

 ## Requirements
 - Python 3.8+
 - Libraries: pandas, numpy, sklearn, xgboost, matplotlib, seaborn, joblib, streamlit

 ## Team
 [Your Group Member Names and IDs]

 ## License
 MIT License
 """
    with open('outputs/README.md', 'w') as f:
        f.write(readme_content)

    # Generate model details file
    model_details = """# Model Creation Details

 ## Model Selection
 XGBoost was chosen due to its:
 - Robustness with tabular data
 - Handling of imbalanced datasets
 - Feature importance insights for identifying key attributes
 - Internal handling of missing values

 ## Preprocessing
 - Created Total_Income, Income_Ratio, and Loan_Term_Ratio features
 - Encoded categorical variables using LabelEncoder
 - Scaled numerical features with StandardScaler
 - Removed outliers using IQR method
 - Handled missing values: mode for categorical, median for numerical
 - Saved preprocessed datasets at each major processing step:
   - data_with_features_pre_encoding.csv: After feature creation but before encoding
   - data_encoded_pre_scaling.csv: After encoding but before scaling
   - full_preprocessed_data.csv: After all preprocessing steps
   - train_data_preprocessed.csv and test_data_preprocessed.csv: Train/test split data

 ## Training
 - Split data: 80% train, 20% test (randomized with random_state=42)
 - Used XGBClassifier with logloss evaluation metric

 ## Tuning
 - GridSearchCV with parameters:
   - n_estimators: [100, 200, 300]
   - max_depth: [3, 5, 7, 9]
   - learning_rate: [0.01, 0.1, 0.3, 0.5]
   - subsample: [0.7, 0.8, 0.9]
   - colsample_bytree: [0.7, 0.8, 0.9]
 - Best parameters saved in `model_performance.txt`

 ## Feature Importance
 - Analyzed feature importance to identify attributes most critical for loan approval
 - Complete feature importance rankings saved in `feature_importance.csv`
 - Top 5 features saved in `feature_importance_scores.txt`
 - Visualization of top 10 features in `outputs/feature_importance.png`

 ## Evaluation
 - Initial and tuned accuracies saved in `model_performance.txt`
 - Visualizations: loan status distribution, applicant income, loan amount vs. status, credit history vs. status, confusion matrix, feature importance

 ## Saved Artifacts
 - label_encoders.pkl: Label encoders for categorical variables
 - numerical_scaler.pkl: StandardScaler for numerical variables
 - loan_prediction_model.pkl: Trained and tuned XGBoost model
 """
    with open('outputs/model_details.txt', 'w') as f:
        f.write(model_details)

    return best_model, initial_accuracy, tuned_accuracy, grid_search.best_params_, top_features


# Load model and other necessary data
def load_model_data():
    try:
        best_model = joblib.load('outputs/loan_prediction_model.pkl')
        with open('outputs/model_performance.txt', 'r') as f:
            lines = f.readlines()
            initial_accuracy = float(lines[0].split(': ')[1])
            tuned_accuracy = float(lines[1].split(': ')[1])
            best_params = lines[2].strip()

        with open('outputs/feature_importance_scores.txt', 'r') as f:
            lines = f.readlines()
            top_features = {}
            for i in range(1, len(lines)):
                feature, score = lines[i].strip().split(': ')
                top_features[feature] = float(score)

        return best_model, initial_accuracy, tuned_accuracy, best_params, top_features
    except:
        # If model doesn't exist yet, train and save it
        return train_and_save_model()


# Task 6: Enhanced Dashboard
def run_dashboard():
    # Apply custom CSS
    apply_custom_css()

    # Load model and data
    best_model, initial_accuracy, tuned_accuracy, best_params, top_features = load_model_data()

    # Title banner
    st.markdown("""
      <div class="title-banner">
          <h1>Motz Financial Services</h1>
          <h2>Loan Prediction Dashboard</h2>
      </div>
      """, unsafe_allow_html=True)

    # Dashboard layout
    st.markdown("""
      <div>
          This dashboard presents the results of a machine learning model built to predict loan approval outcomes
          for Motz Financial Services. The model uses applicant data such as gender, income, and credit history.
      </div>
      """, unsafe_allow_html=True)

    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Insights", "🔮 Prediction Tool", "📈 Visualizations", "🔍 Data Explorer"])

    with tab1:
        # Model metrics section
        st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
              <div class="metric-container">
                  <div class="metric-label">Initial Accuracy</div>
                  <div class="metric-value">{:.2%}</div>
              </div>
              """.format(initial_accuracy), unsafe_allow_html=True)

        with col2:
            st.markdown("""
              <div class="metric-container">
                  <div class="metric-label">Tuned Accuracy</div>
                  <div class="metric-value">{:.2%}</div>
              </div>
              """.format(tuned_accuracy), unsafe_allow_html=True)

        with col3:
            st.markdown("""
              <div class="metric-container">
                  <div class="metric-label">Improvement</div>
                  <div class="metric-value">{:.2%}</div>
              </div>
              """.format(tuned_accuracy - initial_accuracy), unsafe_allow_html=True)

        # Feature importance section
        st.markdown("<h3>Most Important Features for Loan Approval</h3>", unsafe_allow_html=True)
        st.markdown("<p>The following features have the greatest impact on loan approval decisions:</p>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            # Create a horizontal bar chart for feature importance
            feature_names = list(top_features.keys())
            feature_values = list(top_features.values())

            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            y_pos = np.arange(len(feature_names))

            # Horizontal bar chart with color gradient
            bars = ax.barh(y_pos, feature_values, color=plt.cm.Blues(np.linspace(0.6, 1, len(feature_names))))

            # Add feature names and values
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance for Loan Approval')

            # Add value labels to the bars
            for i, v in enumerate(feature_values):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            for feature, score in top_features.items():
                st.markdown(f"""
                  <div class="feature-item">
                      <strong>{feature}</strong>: {score:.4f}
                  </div>
                  """, unsafe_allow_html=True)

        # Adding confusion matrix to model insights tab
        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.image('outputs/confusion_matrix.png', caption='Confusion Matrix', use_container_width=True)

    with tab2:
        st.markdown("<h3>Predict Loan Approval</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='column-container'>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='column-container'>", unsafe_allow_html=True)
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount (thousands)", min_value=0, value=150)
            loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0, value=360)
            credit_history = st.selectbox("Credit History", [1, 0],
                                          format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)")
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Predict Loan Approval"):
            # Preparing input data
            input_data = pd.DataFrame({
                'Gender': [1 if gender == "Male" else 0],
                'Married': [1 if married == "Yes" else 0],
                'Dependents': [3 if dependents == "3+" else int(dependents)],
                'Education': [1 if education == "Graduate" else 0],
                'Self_Employed': [1 if self_employed == "Yes" else 0],
                'ApplicantIncome': [applicant_income],
                'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount],
                'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [credit_history],
                'Property_Area': [2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0],
                'Total_Income': [applicant_income + coapplicant_income],
                'Income_Ratio': [applicant_income / (
                        applicant_income + coapplicant_income) if applicant_income + coapplicant_income > 0 else 0],
                'Loan_Term_Ratio': [loan_amount / loan_amount_term if loan_amount_term > 0 else 0]
            })

            # Get scaler for numerical columns
            scaler = joblib.load('outputs/numerical_scaler.pkl')
            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income',
                              'Income_Ratio', 'Loan_Term_Ratio']

            # Transform numerical columns
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

            # Make prediction
            prediction = best_model.predict(input_data)[0]

            # Display prediction with styling
            if prediction == 1:
                st.markdown("""
                  <div class="prediction-approved">
                      <h3>✅ Loan Approved!</h3>
                      <p>Based on the provided information, the loan is likely to be approved.</p>
                  </div>
                  """, unsafe_allow_html=True)
            else:
                st.markdown("""
                  <div class="prediction-rejected">
                      <h3>❌ Loan Rejected</h3>
                      <p>Based on the provided information, the loan is likely to be rejected.</p>
                  </div>
                  """, unsafe_allow_html=True)

            # Probability score
            proba = best_model.predict_proba(input_data)[0]
            approval_probability = proba[1] if len(proba) > 1 else proba[0]

            # Create gauge chart for probability
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)

            # Create the gauge
            gauge_colors = [(0.8, 0.2, 0.2), (0.8, 0.8, 0.2), (0.2, 0.8, 0.2)]  # R, Y, G

            # Draw the gauge background
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.2, color='lightgray', alpha=0.5))

            # Draw the gauge value
            if approval_probability <= 0.33:
                color = gauge_colors[0]
            elif approval_probability <= 0.66:
                color = gauge_colors[1]
            else:
                color = gauge_colors[2]

            ax.add_patch(plt.Rectangle((0, 0), approval_probability, 0.2, color=color))

            # Add ticks and labels
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.3)
            ax.set_xticks([0, 0.33, 0.66, 1])
            ax.set_xticklabels(['0%', '33%', '66%', '100%'])
            ax.set_yticks([])

            # Add a title
            ax.set_title(f'Approval Probability: {approval_probability:.2%}')

            # Display the gauge
            st.pyplot(fig)

            # Add explanation of factors
            st.markdown("<h4>Key Factors Affecting This Decision:</h4>", unsafe_allow_html=True)

            explanation = []

            # Credit history is typically very important
            if credit_history == 0:
                explanation.append("Poor credit history significantly reduces approval chances")
            else:
                explanation.append("Good credit history improves approval chances")

            # Income related factors
            total_income = applicant_income + coapplicant_income
            if total_income < 3000:
                explanation.append("Lower income may affect loan approval")
            elif total_income > 10000:
                explanation.append("Higher income improves approval chances")

            # Loan amount to income ratio
            loan_to_income = loan_amount / (total_income if total_income > 0 else 1)
            if loan_to_income > 0.5:
                explanation.append("High loan amount compared to income")
            else:
                explanation.append("Reasonable loan amount compared to income")

            # Display explanations
            for i, exp in enumerate(explanation):
                st.markdown(f"""
                <div style="padding: 0.5rem; margin-bottom: 0.5rem; background-color: {'#f1f8fe' if i % 2 == 0 else '#f8f9fa'}; 
                         border-left: 3px solid {'#3498db' if i % 2 == 0 else '#2c3e50'}; border-radius: 4px;">
                    • {exp}
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h3>Data Visualizations</h3>", unsafe_allow_html=True)

        # Check if visualization files exist, if not create them
        viz_directory = "outputs"
        if not os.path.exists(f"{viz_directory}/loan_status_distribution.png"):
            # Create visualizations
            df = load_and_clean_data()

            if not os.path.exists(viz_directory):
                os.makedirs(viz_directory)

            # Loan Status Distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Loan_Status', data=df)
            plt.title('Loan Status Distribution')
            plt.savefig(f'{viz_directory}/loan_status_distribution.png')
            plt.close()

            # Applicant Income Distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(df['ApplicantIncome'], kde=True)
            plt.title('Applicant Income Distribution')
            plt.savefig(f'{viz_directory}/applicant_income_distribution.png')
            plt.close()

            # Loan Amount vs. Loan Status
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)
            plt.title('Loan Amount vs. Loan Status')
            plt.savefig(f'{viz_directory}/loan_amount_vs_status.png')
            plt.close()

            # Credit History vs. Loan Status
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
            plt.title('Credit History vs. Loan Status')
            plt.savefig(f'{viz_directory}/credit_history_vs_status.png')
            plt.close()

        # Display visualizations
        viz_tabs = st.tabs([
            "Loan Status Distribution",
            "Applicant Income",
            "Loan Amount vs Status",
            "Credit History Impact"
        ])

        with viz_tabs[0]:
            st.image(f'{viz_directory}/loan_status_distribution.png', use_container_width=True)
            st.markdown("""
            <p>This chart shows the distribution of approved vs. rejected loans in the dataset. 
            Understanding this distribution helps us assess if the dataset is balanced.</p>
            """, unsafe_allow_html=True)

        with viz_tabs[1]:
            st.image(f'{viz_directory}/applicant_income_distribution.png', use_container_width=True)
            st.markdown("""
            <p>This histogram displays the distribution of applicant incomes. 
            The distribution is right-skewed, indicating that most applicants have lower to middle incomes 
            with fewer high-income applicants.</p>
            """, unsafe_allow_html=True)

        with viz_tabs[2]:
            st.image(f'{viz_directory}/loan_amount_vs_status.png', use_container_width=True)
            st.markdown("""
            <p>This boxplot compares loan amounts between approved and rejected applications. 
            It helps identify if there's a clear pattern between loan amount and approval status.</p>
            """, unsafe_allow_html=True)

        with viz_tabs[3]:
            st.image(f'{viz_directory}/credit_history_vs_status.png', use_container_width=True)
            st.markdown("""
            <p>This chart shows the relationship between credit history and loan approval status. 
            Credit history (1 = good history, 0 = bad history) has a significant impact on loan approval.</p>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Motz Financial Services | Loan Prediction System | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_dashboard()