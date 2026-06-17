import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

# ==========================
# DATA CLEANING
# ==========================

# before cleaning
print("Shape before cleaning:", df.shape)
#finding missing values
print("Missing values in each column:")
print(df.isnull().sum())

print("\nUnique values in categorical columns:")
print("loan_intent:", df['loan_intent'].unique())
print("loan_grade:", df['loan_grade'].unique())
print("cb_person_default_on_file:", df['cb_person_default_on_file'].unique())

print("\nAnomaly check - age over 100:")
print(df[df['person_age'] > 100])

print("\nAnomaly check - employment length over 60:")
print(df[df['person_emp_length'] > 60])

# Removing impossible ages
df = df[df['person_age'] <= 100]

# Removing impossible employment lengths
df = df[df['person_emp_length'] <= 60]

# Filling missing employment length with median
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# Filling missing interest rate with median
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

print("Shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# to save cleaned data
df.to_csv('credit_risk_cleaned.csv', index=False)
print("\nCleaned data saved!")

# ==========================
# FEATURE ENGINEERING
# ==========================

# Loan to Income Ratio - how much loan vs what they earn
df['loan_to_income'] = df['loan_amnt'] / df['person_income']

# Monthly payment estimate
df['monthly_payment'] = (df['loan_amnt'] * df['loan_int_rate'] / 100) / 12

# Payment to Income Ratio - can they afford monthly payments?
df['payment_to_income'] = df['monthly_payment'] / (df['person_income'] / 12)

# Credit history to age ratio - how long have they had credit vs their age
df['credit_to_age'] = df['cb_person_cred_hist_length'] / df['person_age']

print("New features created:")
print(df[['loan_to_income', 'monthly_payment', 'payment_to_income', 'credit_to_age']].head(10))

df.to_csv('credit_risk_features.csv', index=False)
print("\nFeature dataset saved!")

# ==========================
# EXPLORATORY RISK ANALYSIS
# ==========================


