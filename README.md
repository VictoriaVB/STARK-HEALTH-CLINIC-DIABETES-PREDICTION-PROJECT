Diabetes Prediction Model

Background

Stark Health Clinic is a leading healthcare provider dedicated to enhancing patient care through technology and predictive modeling. By integrating machine learning into its operations, the clinic aims to improve disease detection, optimize resource allocation, and drive better health outcomes.

Diabetes presents a major health challenge for Stark Health’s patients, carrying significant medical and financial implications. Current early detection methods lack precision, resulting in missed opportunities for timely intervention. To address this, the clinic seeks to develop an advanced diabetes prediction model that enables proactive patient management and reduces long-term healthcare costs.

Objective

The goal of this project is to develop a robust machine learning model that accurately predicts an individual's risk of developing diabetes. By analyzing patient data and identifying key risk factors, the model will support early detection efforts, allowing for targeted preventive measures and improved clinical decision-making.

Approach

As a Data Scientist leading this initiative, the project involves:

Data Collection & Preprocessing: Aggregating patient data, including demographics, medical history, lifestyle factors, and laboratory results, ensuring data quality and compliance.

Feature Engineering: Identifying and selecting the most relevant predictors for diabetes risk.

Model Development: Implementing machine learning algorithms such as Decision Trees, Random Forest, Gradient Boosting, and Neural Networks to develop an accurate predictive model.

Model Evaluation & Validation: Assessing model performance using key metrics (e.g., accuracy, precision, recall, F1-score) and fine-tuning hyperparameters for optimal results.

Deployment & Integration: Collaborating with clinical teams to integrate the model into Stark Health’s healthcare system, enabling real-time predictions and actionable insights.

Continuous Monitoring & Improvement: Establishing a framework for ongoing model monitoring, retraining, and refinement based on real-world data.

Expected Impact

By implementing this diabetes risk prediction model, Stark Health Clinic aims to:

Enhance Early Detection: Improve the accuracy of diabetes diagnosis at an earlier stage.

Optimize Patient Outcomes: Enable timely interventions and personalized treatment plans.

Reduce Healthcare Costs: Minimize long-term financial burdens associated with diabetes complications.

Support Proactive Healthcare: Empower medical professionals with data-driven insights to take preventive action.

This initiative represents a significant step toward transforming diabetes management through AI-driven predictive analytics, reinforcing Stark Health Clinic’s commitment to innovation and excellence in patient care.

Installation & Usage (Optional Section)

If this project includes code, add instructions on how to install dependencies and run the model.
### For data analysis 
import pandas as pd 
import numpy as np

### For data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

from collections import Counter



#scikit-learn s
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification

from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm  import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# Loading the dataset 
df=pd.read_csv("C:\\Users\\vicky\\Downloads\\diabetes_prediction_dataset - diabetes_prediction_dataset.csv")


df.head(2)#  for viewing the top two rows

df.tail(2)#  for viewing the bottom  ten rows

df.sample(5)# checking out random five rows

### Dimensionality to explore no of rows and colum
df.shape

print("Number of columns",df.shape[0])
print("Number of rows",df.shape[1])

###  Checking the 9 features of the data
df.columns

### Investigating the data set for abnomalies 
df.info()

### Numerical statiscal Analysis
df.describe ()

### Checking for the missing values
print(df.isnull().sum())
 ### Visualisizing  the missing values 
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=True ,cmap="magma_r");


#### The uniform color suggests that all data points across all columns (e.g., gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level, and diabetes) are fully available.

### Outlier Analysis

### Checking for outliers 
fig,axs=plt.subplots(nrows=2, ncols=2, figsize=(15,10))
sns.boxplot(x="blood_glucose_level", data=df, ax=axs[0,0])
axs[0,0].set_title ("Boxplot on age ")

sns.boxplot(x="bmi", data=df, ax=axs[0,1])
axs[0,1].set_title ("Boxplot on bmi ")

sns.boxplot(x="age", data=df, ax=axs[1,0])
axs[1,0].set_title ("Boxplot on age ")

sns.boxplot(x="HbA1c_level", data=df, ax=axs[1,1])
axs[1,1].set_title ("Boxplot on HbA1c_level ");

 #### BMI and Blood Glucose Level have significant outliers, indicating potential high-risk patients. Age and HbA1c Level are more normally distributed with fewer extremes. Further analysis is needed to assess the impact of BMI and blood glucose outliers on diabetes prediction.



## Univariate Analysis 

### Number of female: male in the gender column
plt.figure(figsize=(8,5))
ax= sns.countplot(x=df["gender"],order= df["gender"].value_counts(ascending=False).index)
values=df["gender"].value_counts(ascending=False).values
plt.xlabel("Gender")
plt.ylabel("Count of patient")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

#### It has been observed that the majority of patients are female compared to male and we have an insignificant numbers that are others.

### Number of patients with hypertension  
def hypertensionlabel2(tg):
    if tg == 1:
        return "Yes"
    else:
        return "No"
    
df["hypertensionlabel2"] = df["hypertension"].apply(hypertensionlabel2)


plt.figure(figsize=(8,4))
ax= sns.countplot(x=df['hypertensionlabel2'],order= df['hypertensionlabel2'].value_counts(ascending=False).index)
values=df['hypertensionlabel2'].value_counts(ascending=False).values
plt.xlabel('Hypertension')
plt.ylabel("Count of patient hypertension")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

####  The majority of patients have not had hypertension.

### Number of patients with heart disease  


def heartdislabel3(tg):
    if tg == 1:
        return "Yes"
    else:
        return "No"
    
df["heartdislabel3"] = df['heart_disease'].apply(heartdislabel3)

plt.figure(figsize=(8,4))
ax= sns.countplot(x=df["heartdislabel3"],order= df["heartdislabel3"].value_counts(ascending=False).index)
values=df["heartdislabel3"].value_counts(ascending=False).values
plt.xlabel('heart_disease')
plt.ylabel("Count of patient heart_disease")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

####  The majority of patients have not had heart disease.

### Number of patients with diabetes

def diabeteslabel4(tg):
    if tg == 1:
        return "Yes"
    else:
        return "No"
    
df["diabeteslabel4"] = df['diabetes'].apply(diabeteslabel4)
plt.figure(figsize=(8,4))
ax= sns.countplot(x=df['diabeteslabel4'],order= df['diabeteslabel4'].value_counts(ascending=False).index)
values=df['diabeteslabel4'].value_counts(ascending=False).values
plt.xlabel('diabetes')
plt.ylabel("Count of patient diabetes")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

####  The majority of patients have not had diabetes which is the main target of our study.

### Number of patients with Smoking History  
plt.figure(figsize=(8,4))
ax= sns.countplot(x=df['smoking_history'],order= df['smoking_history'].value_counts(ascending=False).index)
values=df['smoking_history'].value_counts(ascending=False).values
plt.xlabel('smoking_history')
plt.ylabel("Count of patient smoking_history")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

####  We lack information on the smoking history for a considerable portion of patients, and a significant number either do not smoke or have quit.

### For grouping the age into various age groups 
def age_group(x):
    if x <= 25:
        return "Youth Adult(25-39)"
    elif x <= 40:
        return " Adult(40-49)"
    elif x <= 50:
        return " Older Adult(50-64)"
    else:
        return "Elder(65 and above)"

### Adding the age group  to the table 
df ["age_group"] = df ["age"].apply(age_group)
        
df.head(2)# viewing the table 

### Visualizing the age group 
plt.figure(figsize=(15,5))
ax= sns.countplot(y=df["age_group"], data=df,order= df["age_group"].value_counts(ascending=False).index)
values=df["age_group"].value_counts(ascending=False).values
plt.xlabel("Age_group")
plt.ylabel("Count of Age_group")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

def bmi_group(x):
    if x <= 18.5:
        return "Underweight"
    elif x <= 24.9:
        return "Healthy weight"
    elif x <= 29.9:
        return "Overweight"
    else:
        return "Obesity "

df ["bmi_group"] = df ["bmi"].apply(bmi_group)
        
df.head(2)

### Visualizing the bmi group 
plt.figure(figsize=(15,5))
bx= sns.countplot(y=df["bmi_group"],order= df["bmi_group"].value_counts(ascending=False).index)
values=df["bmi_group"].value_counts(ascending=False).values
plt.xlabel("Count of BMI_group")
plt.ylabel("BMI_group")
plt.title("Total Number of Patients")
bx.bar_label(container=bx.containers[0],labels=values);

def HbA1c_group(x):
    if x <= 6.0:
        return "Normal"
    elif x <= 6.4:
        return "Prediabetes"
    else:
        return ">= 6.5 Diabetes "

df [" HbA1c_group"] = df ["HbA1c_level"].apply( HbA1c_group)
        
df.head(2)

### Customer gender
plt.figure(figsize=(15,5))
Hx= sns.countplot(y=df[" HbA1c_group"],order= df[" HbA1c_group"].value_counts(ascending=False).index)
values=df[" HbA1c_group"].value_counts(ascending=False).values
plt.xlabel("Count of HbA1c_group")
plt.ylabel("HbA1c_group")
plt.title("Total Number of Patients")
Hx.bar_label(container=Hx.containers[0],labels=values);

#### In this demographic, individuals aged 65 and above, classified as elderly, are the most prevalent, followed by young adults. Additionally, there is a higher prevalence of overweight and obese individuals, with the majority also having normal HbA1C levels.

### Bivariate Analysis 

### Number of female: male in the gender column
plt.figure(figsize=(8,5))
ax= sns.countplot(x=df["gender"],order= df["gender"].value_counts(ascending=False).index)
values=df["gender"].value_counts(ascending=False).values
plt.xlabel("Gender")
plt.ylabel("Count of patient gender")
plt.title("Total Number of Patients")
ax.bar_label(container=ax.containers[0],labels=values);

plt.figure(figsize = (15,7))
ag=sns.countplot(x='age_group', data=df, hue= 'diabeteslabel4' )
plt.xlabel('Age Group')
plt.ylabel('Count of age_group')
plt.title('Total Number of Patients')
for container in ag.containers:
    ag.bar_label(container, labels=[f'{int(height.get_height())}' for height in container], padding=5)
plt.show()

#label4 is the relabbelled diabetes with yes and no rather than 0 and 1

plt.figure(figsize = (15,7))
gx=sns.countplot(x='gender', data=df, hue= 'diabeteslabel4' )
plt.xlabel('gender')
plt.ylabel('Count of gender')
plt.title('Total Number of Patients')
### This code ensures conversion to int only happens for valid numbers.
### If NaN is found, it replaces the label with an empty string '' to avoid errors.to show the number label

for container in gx.containers:
    gx.bar_label(container, 
                 labels=[f'{int(height.get_height())}' if not np.isnan(height.get_height()) else '' for height in container], 
                 padding=5)

plt.show()

plt.figure(figsize = (15,7))
hy=sns.countplot(x='hypertensionlabel2', data=df, hue= 'diabeteslabel4' )
plt.xlabel('hypertension')
plt.ylabel('Count of hypertension')
plt.title('Total Number of Patients')
for container in hy.containers:
    hy.bar_label(container, labels=[f'{int(height.get_height())}' for height in container], padding=5)
plt.show()

plt.figure(figsize = (15,7))
hd=sns.countplot(x='heartdislabel3', data=df, hue= 'diabeteslabel4' )
plt.xlabel('heart_disease')
plt.ylabel('Count of heart_disease')
plt.title('Total Number of Patients')
for container in hd.containers:
    hd.bar_label(container, labels=[f'{int(height.get_height())}' for height in container], padding=5)
plt.show()

plt.figure(figsize = (15,7))
sh=sns.countplot(x='smoking_history', data=df, hue= 'diabeteslabel4' )
plt.xlabel('smoking_history')
plt.ylabel('Count of smoking_history')
plt.title('Total Number of Patients')
for container in sh.containers:
    sh.bar_label(container, labels=[f'{int(height.get_height())}' for height in container], padding=5)
plt.show()

plt.figure(figsize = (15,7))
bg=sns.countplot(x='bmi_group', data=df, hue= 'diabeteslabel4' )
plt.xlabel('bmi_group')
plt.ylabel('Count of bmi_group')
plt.title('Total Number of Patients')
for container in bg.containers:
    bg.bar_label(container, labels=[f'{int(height.get_height())}' for height in container], padding=5)
plt.show()

### Multivariate Analysis

df.columns

### Grouping and aggregating numeric columns correctly
procat1 = df.groupby('diabeteslabel4')[['bmi', 'age']].sum().reset_index()

### Counting gender occurrences separately (assuming gender is categorical)
gender_counts = df.groupby('diabeteslabel4')['gender'].value_counts().unstack().reset_index()

### Reshaping procat1 using melt
procat1 = pd.melt(procat1, id_vars='diabeteslabel4', var_name='Metric', value_name="Total")

### Plotting
plt.figure(figsize=(8, 5))
sns.barplot(data=procat1, x='diabeteslabel4', y="Total", hue='Metric')
plt.title("Comparison of BMI and Age by Diabetes Status")
plt.show()

### Grouping and aggregating numeric columns correctly
procat1 = df.groupby('diabeteslabel4')[['bmi', 'age']].sum().reset_index()

### Counting gender occurrences separately (assuming gender is categorical)
gender_counts = df.groupby('diabeteslabel4')['gender'].value_counts().unstack().reset_index()

### Reshaping procat1 using melt
procat1 = pd.melt(procat1, id_vars='diabeteslabel4', var_name='Metric', value_name="Total")

### Plotting
plt.figure(figsize=(8, 5))
sns.barplot(data=procat1, x='diabeteslabel4', y="Total", hue='Metric')
plt.title("Comparison of BMI and Age by Diabetes Status")
plt.show()

### Counting genders separately
gender_counts = df.groupby('diabetes')['gender'].value_counts().unstack().reset_index()

### Melting gender counts
gender_melted = pd.melt(gender_counts, id_vars='diabetes', var_name='Metric', value_name='Total')

### Combining both datasets
procat1 = pd.concat([procat1, gender_melted], ignore_index=True)

### Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=gender_melted, x='diabetes', y="Total", hue='Metric')
plt.title("Comparison of BMI, Age, and Gender Counts by Diabetes Status")
plt.show()


gender_melted

### Analyzing 3 or more columns
procat1= df.groupby( 'diabeteslabel4')[[ 'hypertension', 'bmi', 'age','heart_disease']].sum().reset_index()
procat1 = pd.melt(procat1, id_vars= 'diabeteslabel4',var_name ='Metric',value_name="Total")
sns.barplot(data=procat1,x='diabeteslabel4',y="Total",hue='Metric')

procat1





### Corelation 

d = df.corr()
plt.figure(figsize=(15,7))
sns.heatmap(d, vmax=7,square=True,annot=True);

The heatmap shows how various health factors are correlated with diabetes:

Strongest correlations:

Blood Glucose (0.42) and HbA1c (0.40) are the most significant predictors of diabetes risk.

Moderate correlations:

Age (0.26), Hypertension (0.20), BMI (0.21), and Heart Disease (0.17) are all linked to increased diabetes risk, but with weaker correlations.

Interrelationships:

Age & BMI (0.34): Older individuals tend to have higher BMI.

Hypertension & Heart Disease (0.12): Mild correlation, as they often occur together.
Blood Glucose & HbA1c (0.17): Both measure blood sugar at different times.


Blood glucose and HbA1c are the key risk factors, while age, BMI, hypertension, and heart disease also contribute, though less strongly. Managing one factor (e.g., BMI) may reduce the risk of others (e.g., hypertension, diabetes).



### Feature Engineering

df1 = df[['gender','age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level']]

label1= df[['diabetes']]

df1 = df1.copy()  # Create an explicit copy to avoid SettingWithCopyWarning

df1.loc[:, 'gender'] = df1['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})

df2 = df1.copy()  # Create an explicit copy to avoid SettingWithCopyWarning

df2.loc[:, 'smoking_history'] = df2['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2,'former': 3,'not current': 4,'ever': 5})

df2.head(2)

### Machine Learning 

### Normalization it has taken care of the outlier 
scaler = MinMaxScaler()

df2["scalar_"]  = scaler.fit_transform(df2['bmi'].values.reshape(-1,1))
df2["scalar_bmi"]  = scaler.fit_transform(df2['bmi'].values.reshape(-1,1))
df2["scalar_bgl"]  = scaler.fit_transform(df2['blood_glucose_level'].values.reshape(-1,1))
df2["scalar_hba1c"] = scaler.fit_transform(df2['HbA1c_level'].values.reshape(-1,1))
df2["scalar_SH"] = scaler.fit_transform(df2['smoking_history'].values.reshape(-1,1)) 
df2.drop(['bmi','blood_glucose_level','HbA1c_level','smoking_history'],axis =1, inplace= True)

df2.head(2)

### Split Data
X_train, X_test, y_train, y_test = train_test_split(df2, label1, test_size=0.2, random_state=42)


df2.shape

### Train Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

### Predictions
ly_pred =logreg.predict(X_test)

### Print Model Details
print("LogisticRegression")
print("Accuracy:", accuracy_score(y_test, ly_pred))

print("Precision:", precision_score(y_test, ly_pred))
print("Recall: ", recall_score(y_test, ly_pred))
print("F1 Score:", f1_score(y_test, ly_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, ly_pred))

### create a confusion matrix
lcm = confusion_matrix(y_test, ly_pred)

#d = df.corr()
sns.heatmap(lcm, cmap="Blues",square=True,fmt='g', annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

##### Confusion Matrix Analysis The confusion matrix provides insights into the model's classification performance for diabetes prediction. 
The four values represent: True Negatives (TN) = 18,146 → The model correctly predicted non-diabetic cases.False Positives (FP) = 146 → The model incorrectly predicted diabetic when the patient was actually non-diabetic (Type I Error).False Negatives (FN) = 687 → The model incorrectly predicted non-diabetic when the patient was actually diabetic (Type II Error).True Positives (TP) = 1,021 → The model correctly predicted diabetic cases.Findings and Interpretation High True Negative Rate The model performs exceptionally well in detecting non-diabetic cases, with 18,146 correct predictions. Low False Positive Rate (146 cases) Very few non-diabetic individuals were misclassified as diabetic. This is a good outcome because it reduces unnecessary anxiety and medical tests for healthy individuals. Moderate False Negative Rate (687 cases) The model misses some diabetic cases, which is concerning because failing to diagnose diabetes can have severe health consequences. Good True Positive Count (1,021 cases) The model identifies a reasonable number of actual diabetic cases correctly, but there is room for improvement. Conclusion and Next Steps The model performs well overall, with a high accuracy in predicting non-diabetic cases.However, the 687 false negatives indicate that some diabetic cases are being missed. Possible improvements: Adjusting the classification threshold to increase sensitivity. Using a different evaluation metric, such as F1-score or Recall, to balance false negatives.Exploring feature importance to see if additional factors can improve diabetic detection.

### Train Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

### Predictions
r_pred = model.predict(X_test)
print("LogisticRegression")
print("Accuracy:", accuracy_score(y_test, r_pred))

print("Precision:", precision_score(y_test, r_pred))
print("Recall: ", recall_score(y_test, r_pred))
print("F1 Score:", f1_score(y_test, r_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, r_pred))

### Create a confusion matrix
lcm = confusion_matrix(y_test, r_pred)

#d = df.corr()
sns.heatmap(lcm, cmap="Blues",square=True,fmt='g', annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

### 8 Machine learning algorithm
classifiers = [[ XGBClassifier(), "XGBclassifier"],
               [ RandomForestClassifier(), "Random Forest"],
               [KNeighborsClassifier(), "K-Nearest Neighbors"],
               [SGDClassifier(), "SGD Classifier " ],
               [ SVC(), "Support Vector Machine"],
               [GaussianNB(), "Naive Bayes"],
               [DecisionTreeClassifier(), "Decision Tree"],
               [ LogisticRegression(), "Logistic Regression"]]

acc_list ={}
precision_list ={}
recall_list ={}
roc_list ={}

for classifier in classifiers:
    
    model = classifier[0]
    model.fit(X_train,y_train )
    model_name = classifier[1]
    pred = model.predict(X_test) 
    
    a_score= accuracy_score(y_test, pred)
    p_score= precision_score(y_test, pred)
    r_score= recall_score(y_test, pred)
    roc_score= roc_auc_score(y_test, pred)
    
    acc_list[model_name] = ([str(round(a_score*100, 2)) + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    
    if model_name is classifiers[-1][1]:
       print('')
    
    

### Create a confusion matrix
lcm = confusion_matrix(y_test, pred)

#d = df.corr()
sns.heatmap(lcm, cmap="Blues",square=True,fmt='g', annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

acc_list

print("Accuracy Score")
sl = pd.DataFrame(acc_list)
sl.head()


print("Recall Score")
s3 = pd.DataFrame(recall_list)
s3.head()

print("auc Roc Score")
s4 = pd.DataFrame(roc_list)
s4.head()

for classifier in classifiers:
    print(type(classifier[0]), classifier[1])

##### Key Observations
High True Negatives (18,146 cases)

The model correctly predicted non-diabetic individuals with high accuracy.
Low False Positives (146 cases)

Only 146 non-diabetic individuals were incorrectly classified as diabetic.
This means the model has a high specificity (ability to correctly identify non-diabetic cases).
False Negatives (687 cases)

These are actual diabetic cases that the model failed to identify.
This is concerning because missing a diabetic diagnosis can have serious health implications.
True Positives (1,021 cases)

The model correctly identified diabetic cases, but it could be improved to reduce false negatives.


Conclusion & Model Performance
Strengths:
The model demonstrates high accuracy and strong specificity, meaning it effectively identifies non-diabetic individuals with minimal false positives.
Weaknesses:
A relatively high number of false negatives indicates that some diabetic cases are being missed, which is a critical concern in medical applications.
Areas for Improvement:
Enhancing Recall (Sensitivity): The model should be optimized to correctly identify more diabetic cases.
Potential Solutions:
Adjusting the decision threshold to favor recall over precision.
Implementing cost-sensitive learning to penalize misclassification of diabetic cases.
Using ensemble methods to improve overall predictive performance.
Final Thoughts:
While the model performs well overall, the presence of false negatives poses a risk in a healthcare setting where missing a diabetes diagnosis can have severe consequences. Further refinements, such as addressing class imbalance and tuning hyperparameters, can improve its effectiveness in detecting diabetic patients.


Contributor

Victoria Baba – Data Scientist
