Diabetes Prediction Model
Background
Stark Health Clinic integrates machine learning to enhance disease detection, optimize resources, and improve patient outcomes. Diabetes poses significant health and financial challenges, and current detection methods lack precision. This project aims to develop an advanced prediction model for early intervention and cost reduction.

Objective
Develop a machine learning model to predict diabetes risk, enabling early detection and targeted preventive measures.

Methodology
Data Collection & Preprocessing
Data Source: Patient records including demographics, medical history, lifestyle factors, and lab results. Data was anonymized to comply with HIPAA regulations.
Dataset Loading: Used pandas.read_csv for data import.
Feature Engineering: Selected key predictors for diabetes risk.
Model Development
Implemented Decision Trees, Random Forest, Gradient Boosting, and Neural Networks.
Model Evaluation & Validation
Assessed performance using accuracy, precision, recall, and F1-score.
Fine-tuned hyperparameters for optimization.
Deployment & Integration
Integrated into Stark Health’s system for real-time predictions.
Continuous Monitoring & Improvement
Established a framework for ongoing evaluation and retraining.
Expected Impact
Early Detection: Improved accuracy in identifying diabetes risk.
Better Patient Outcomes: Enables timely interventions.
Cost Reduction: Lowers long-term healthcare expenses.
Proactive Healthcare: Empowers clinicians with data-driven insights.
Findings
1. Data Exploration & Cleaning
No Missing Values: Verified using isnull().sum() and a heatmap.
Outlier Detection: Boxplots identified high-risk patients based on blood glucose, BMI, and age.
2. Analysis
Univariate: Most patients are female, elderly, and non-smokers.
Bivariate: Higher diabetes prevalence in older adults and females.
Multivariate: Strong correlation between diabetes and blood glucose, HbA1c, BMI, hypertension, and heart disease.
Correlation Matrix: Blood glucose and HbA1c were the strongest predictors.
3. Model Training & Evaluation
Data Splitting: Used train_test_split.
Tested Models:
Best Performers: Logistic Regression, Random Forest, XGBoost.
Metric Comparison: Analyzed accuracy, precision, recall, and ROC-AUC.
Confusion Matrix: Identified false positives and false negatives.
4. Improvements & Next Steps
Adjust classification thresholds for better recall.
Address class imbalance using SMOTE or weighted loss functions.
Implement ensemble methods for higher accuracy.
Conclusion
Key Insights: Blood glucose and HbA1c are primary diabetes predictors, with age and BMI as contributing factors.
Model Performance: Logistic Regression performed well but needs better recall.
Next Steps: Improve recall, address class imbalance, and refine hyperparameters.
Contributor
Victoria Baba – Data Scientist
