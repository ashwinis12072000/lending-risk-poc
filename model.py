import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('credit_risk_features.csv')

# Convert text columns to numbers
le = LabelEncoder()
df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])
df['loan_intent'] = le.fit_transform(df['loan_intent'])
df['loan_grade'] = le.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = le.fit_transform(df['cb_person_default_on_file'])

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# Build the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nAUC-ROC Score:", round(roc_auc_score(y_test, y_pred), 2))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ==========================
# EXPLORATORY RISK ANALYSIS
# ==========================

print("\nDEFAULT RATE")

default_rate = df['loan_status'].mean() * 100

print(f"Default Rate: {default_rate:.2f}%")


print("\nDEFAULT RATE BY LOAN GRADE")

grade_risk = (
    df.groupby('loan_grade')['loan_status']
      .mean()
      *100
)

print(grade_risk.sort_values(ascending=False))


print("\nDEFAULT RATE BY LOAN INTENT")

intent_risk = (
    df.groupby('loan_intent')['loan_status']
      .mean()
      *100
)

print(intent_risk.sort_values(ascending=False))

df_original = pd.read_csv('credit_risk_features.csv')

print(df_original['loan_intent'].unique())



df['lti_band'] = pd.qcut(
    df['loan_to_income'],
    q=5,
    duplicates='drop'
)

print("\nDEFAULT RATE BY LOAN TO INCOME")

lti_risk = (
    df.groupby('lti_band')['loan_status']
      .mean()
      *100
)

print(lti_risk)