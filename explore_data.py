import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

# Check for missing values
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
