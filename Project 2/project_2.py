import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

heart_disease_uci_df = pd.read_csv('heart_disease_uci.csv')

# EDA Questions:
# 1. General Information
# ● How many rows and columns are in the dataset?

rows = heart_disease_uci_df.shape[0]
columns = heart_disease_uci_df.shape[1]
print(f'There are {rows} rows and {columns} coloumns in the dataset')

# ● Are there any missing values in the dataset?

missing_values = heart_disease_uci_df.isna().sum().sum()
print(f'Yes, there are {missing_values} missing values in the dataset')

# 2. Target Variable Analysis
# ● What is the distribution of the target variable (presence of heart disease: 0 or 1)?

# to show the distribution of the presence of heart disease, we can use the count plot:
sns.countplot(x='num',data=heart_disease_uci_df,color='darkred')
plt.xlabel('Presence Heart Disease (0 means None)')
plt.title('Distribution of the presence of heart disease')
plt.ylabel('Frequency')
plt.show()

# ● How many patients have heart disease, and how many don’t?

no_heart_disease = heart_disease_uci_df['num'].value_counts()[0]

heart_disease_check = heart_disease_uci_df['num'].value_counts()
heart_disease = heart_disease_check.drop(0).sum()

print(f'{no_heart_disease} Patients dont have heart disease')
print(f'{heart_disease} Patients have heart disease')

# 3. Demographic Insights
# ● What is the age range of patients in the dataset?

min_age= heart_disease_uci_df['age'].min()
max_age= heart_disease_uci_df['age'].max()

print(f'The ages of patients in the dataset range from {min_age} to {max_age}')

# ● What is the gender distribution of the patients?

sns.countplot(x='sex',data=heart_disease_uci_df,palette=['deepskyblue','hotpink'])
plt.xlabel('Genders')
plt.title('Gender Distribution of the Patients')
plt.ylabel('Frequency')
plt.show()

# 4. Health Metrics
# ● What are the average and median values of:
# ○ Resting blood pressure (trestbps)?

print(f'The median value of Resting blood pressure is : {heart_disease_uci_df["trestbps"].median().astype(int)}')
print(f'The average value of Resting blood pressure is : {heart_disease_uci_df["trestbps"].mean().astype(int)}')
print('--------------------------------------------------')

# ○ Serum cholesterol (chol)?

print(f'The median value of Serum cholesterol is : {heart_disease_uci_df["chol"].median().astype(int)}')
print(f'The average value of Serum cholesterol is : {heart_disease_uci_df["chol"].mean().astype(int)}')
print('--------------------------------------------------')

# ○ Maximum heart rate (thalch)?

print(f'The median value of Maximum heart rate is : {heart_disease_uci_df["thalch"].median().astype(int)}')
print(f'The average value of Maximum heart rate is : {heart_disease_uci_df["thalch"].mean().astype(int)}')
print('--------------------------------------------------')

# # 5. Categorical Features
# ● How many patients have exercise-induced angina (exang)?

patients_exang = heart_disease_uci_df['exang'].value_counts().iloc[1]
print(f'{patients_exang} patients have exercise-induced angina')

# ● What are the counts of different chest pain types (cp)?

counts_chest_pain_types = heart_disease_uci_df['cp'].value_counts().reset_index()
counts_chest_pain_types.columns = ['Chest Pain Types', 'Counts']
print(counts_chest_pain_types)

# 6. Visual Analysis
# ● Plot the age distribution of the patients.

sns.histplot(x='age',color='teal',data=heart_disease_uci_df,kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of the Patients')
plt.show()

# ● Compare the average cholesterol levels between patients with and without heart disease

sns.catplot(x='num',y='chol',hue='num',data=heart_disease_uci_df,palette='magma_r',kind='box')
plt.xlabel('Presence of Heart Disease (0 means None)')
plt.ylabel('Cholesterol Levels')
plt.title('Cholesterol Levels between Patients with and without Heart Disease')
plt.show()

# 7. Outliers
# ● Are there any outliers in cholesterol (chol) or resting blood pressure (trestbps)?

# Firstly checking the outliers for cholestrol
q1_chol = heart_disease_uci_df['chol'].quantile(0.25)
q3_chol = heart_disease_uci_df['chol'].quantile(0.75)
iqr = q3_chol - q1_chol
chol_outliers = heart_disease_uci_df[(heart_disease_uci_df['chol'] < (q1_chol - 1.5 * iqr)) | (heart_disease_uci_df['chol'] > (q3_chol + 1.5 * iqr))].count()['chol']
print(f'There are {chol_outliers} outliers in cholestrol')

# and now checking the outliers for resting blood pressure

q1_trestbps = heart_disease_uci_df['trestbps'].quantile(0.25)
q3_trestbps = heart_disease_uci_df['trestbps'].quantile(0.75)
iqr = q3_trestbps - q1_trestbps
trestbps_outliers = heart_disease_uci_df[(heart_disease_uci_df['trestbps'] < (q1_trestbps - 1.5* iqr)) | (heart_disease_uci_df['trestbps'] > (q3_trestbps + 1.5 * iqr))].count()['trestbps']
print(f'There are {trestbps_outliers} outliers in resting blood pressure')