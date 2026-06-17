import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

print("Shape before cleaning:", df.shape)

# Remove impossible ages
df = df[df['person_age'] <= 100]

# Remove impossible employment lengths
df = df[df['person_emp_length'] <= 60]

# Fill missing employment length with median
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# Fill missing interest rate with median
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

print("Shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned data
df.to_csv('credit_risk_cleaned.csv', index=False)
print("\nCleaned data saved!")
