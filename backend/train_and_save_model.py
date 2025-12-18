import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import joblib

print("Loading data...")
df = pd.read_csv('../data/train.csv')

print("Preprocessing data...")
df.columns = df.columns.str.strip()

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

df = pd.get_dummies(df, columns=[
    'workclass', 'education', 'marital-status',
    'occupation', 'race', 'relationship', 'native-country'
], drop_first=True)

num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(columns=['income_>50K'])
y = df['income_>50K']

print(f"Total features: {X.shape[1]}")
print(f"Feature columns: {list(X.columns)}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Training CatBoost model with optimal hyperparameters...")
best_catboost = CatBoostClassifier(
    iterations=362,
    learning_rate=0.10017949745755012,
    depth=7,
    l2_leaf_reg=5.703277993521533,
    border_count=234,
    random_strength=0.08892665394913246,
    bagging_temperature=0.6494508318048355,
    random_state=42,
    verbose=False
)

best_catboost.fit(X_train_resampled, y_train_resampled)

print("Evaluating on test set...")
y_pred_proba = best_catboost.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.46).astype(int)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nSaving model and preprocessing objects...")
joblib.dump(best_catboost, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')

print("\nSaved files:")
print("- model.pkl (CatBoost model)")
print("- scaler.pkl (StandardScaler)")
print("- label_encoder.pkl (LabelEncoder for gender)")
print("- feature_columns.pkl (feature column names)")
print("\nModel training and saving completed successfully!")
