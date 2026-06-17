import pandas as pd

df = pd.read_csv('credit_risk_cleaned.csv')

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
