import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

print("Shape of dataset:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())
