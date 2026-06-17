import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import time

# ── Load & Prepare Data ────────────────────────────────────────
df = pd.read_csv('credit_risk_features.csv')

le = LabelEncoder()
df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])
df['loan_intent'] = le.fit_transform(df['loan_intent'])
df['loan_grade'] = le.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = le.fit_transform(df['cb_person_default_on_file'])

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows:  {X_test.shape[0]}")
print("=" * 55)

# ── Model 1: Random Forest ─────────────────────────────────────
print("\n🌲 Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = round(time.time() - start, 2)

rf_pred = rf_model.predict(X_test)
rf_accuracy = round(accuracy_score(y_test, rf_pred) * 100, 2)
rf_auc = round(roc_auc_score(y_test, rf_pred), 4)

print(f"  Accuracy : {rf_accuracy}%")
print(f"  AUC-ROC  : {rf_auc}")
print(f"  Time     : {rf_time}s")

# ── Model 2: XGBoost ───────────────────────────────────────────
print("\n⚡ Training XGBoost...")
start = time.time()
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_time = round(time.time() - start, 2)

xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = round(accuracy_score(y_test, xgb_pred) * 100, 2)
xgb_auc = round(roc_auc_score(y_test, xgb_pred), 4)

print(f"  Accuracy : {xgb_accuracy}%")
print(f"  AUC-ROC  : {xgb_auc}")
print(f"  Time     : {xgb_time}s")

# ── Comparison Table ───────────────────────────────────────────
print("\n" + "=" * 55)
print("           MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Metric':<20} {'Random Forest':>15} {'XGBoost':>15}")
print("-" * 55)
print(f"{'Accuracy':<20} {rf_accuracy:>14}% {xgb_accuracy:>14}%")
print(f"{'AUC-ROC':<20} {rf_auc:>15} {xgb_auc:>15}")
print(f"{'Training Time':<20} {rf_time:>14}s {xgb_time:>14}s")
print("=" * 55)

# ── Winner ─────────────────────────────────────────────────────
if xgb_auc > rf_auc:
    print("\n✅ XGBoost wins on AUC-ROC!")
elif rf_auc > xgb_auc:
    print("\n✅ Random Forest wins on AUC-ROC!")
else:
    print("\n🤝 Both models tied on AUC-ROC!")

# ── Detailed XGBoost Report ────────────────────────────────────
print("\n📊 XGBoost Detailed Report:")
print(classification_report(y_test, xgb_pred))