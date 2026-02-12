"""
Gym Membership Churn Prediction

Predicts whether a gym member will churn (cancel) next month.
Goal: Identify key drivers of churn using real fitness membership data.

Dataset: Gym Customer Features and Churn (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score


# Data
df_raw = pd.read_csv("gym_churn/gym_churn_us.csv")  

# Fixing output terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Preview of the data
print('First 5 rows:')
print(df_raw.head())
print('\nData info:')
print(df_raw.info())
print('Class balance:\n')

print(df_raw['Contract_period'].value_counts())
print(df_raw['Age'].value_counts())
print(df_raw['Avg_additional_charges_total'].value_counts())
print(df_raw['Month_to_end_contract'].value_counts())
print(df_raw['Avg_class_frequency_total'].value_counts())
print(df_raw['Avg_class_frequency_current_month'].value_counts())

# Feature Engineering Model
X = df_raw.drop(['Churn', 'Group_visits', 'Partner', 'gender'], axis=1)
y = df_raw['Churn']

# Standardizing Numeric Columns
numeric_cols = ['Contract_period',
                'Age',
                'Avg_additional_charges_total',
                'Month_to_end_contract',
                'Avg_class_frequency_total',
                'Avg_class_frequency_current_month'
]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)

# List of Feature names
feature_list = list(X.columns)

# Standardize Numeric Columns
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Logistic regression model
lr = LogisticRegression(max_iter=500)

# Recursive feature elimination
rfe = RFE(estimator=lr, n_features_to_select=7)
rfe.fit(X_train_scaled, y_train)

# Top 7 Recursive Features
selected_features = X_train.columns[rfe.support_].tolist()

# Bar plot of selected features
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), [1]*len(selected_features), color='skyblue')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Selected (1 = Yes)')
plt.title('RFE Selected Features')
plt.tight_layout()
#plt.savefig("rfe_selected_features_gym_churn.png", dpi=300, bbox_inches='tight')
plt.show()


# Accuracy of the model with RFE selected features
X_train_selected_scaled = X_train_scaled[selected_features]
X_test_selected_scaled = X_test_scaled[selected_features]

lr.fit(X_train_selected_scaled, y_train) 
rfe_y_pred = lr.predict(X_test_selected_scaled)
rfr_accuracy = accuracy_score(y_test, rfe_y_pred)
print('\nRecursive feature accuracy score (on test):')
print(rfr_accuracy)
print('\n')

# Running Random Forest Classifier
X_selected_train = X_train_scaled[selected_features]
X_selected_test = X_test_scaled[selected_features]

# Finding best classifier
rf_selected_classifier = {
    'Random Forest Classifier':RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Logistic Regression Classifier': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    'XGBoost Classifier': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight = sum(y_train == 0) / sum(y_train == 1))
}

for name,clf in rf_selected_classifier.items():
    clf.fit(X_selected_train, y_train)
    y_pred = clf.predict(X_selected_test)
    y_pred_proba = clf.predict_proba(X_selected_test)[:, 1]
    accuracy=accuracy_score(y_test,y_pred)
    print(f"{name}")
    print(f"{accuracy*100}")
    print(classification_report(y_test,y_pred))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'\nROC-AUC {name}: {roc_auc:.4f}')
    
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'\nConfusion Matrix {name}:')
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    #plt.savefig(f'{name.replace(' ', '_')}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance plot (for Random Forest only)
    if name == 'XGBoost Classifier':
        importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances.head(10))
        plt.title('Top 7 Feature Importances - XGBoost (after RFE)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_gym_churn.png', dpi=300, bbox_inches='tight')
        plt.show()

