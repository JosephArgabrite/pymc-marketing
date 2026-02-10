"""
Auto Insurance Analysis / Premium Risk Modeling

Analyzing personal auto insurance data to predict premiums, claim risk, or loss ratios.
Goal: Show domain-relevant insights (e.g., factors driving high premiums/claims) using my insurance background.

Dataset: Personal Auto Line of Business (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data
df_raw = pd.read_csv("auto_insurance/synthetic_insurance_data.csv")  

# Fixing output terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Preview of the data
print('First 5 rows:')
print(df_raw.head())
print('\nData info:')
print(df_raw.info())
print('Class balance:\n')
print(df_raw['Conversion_Status'].value_counts(normalize=True))

print(df_raw['Marital_Status'].value_counts())
print(df_raw['Prior_Insurance'].value_counts())
print(df_raw['Claims_Severity'].value_counts())
print(df_raw['Policy_Type'].value_counts())
print(df_raw['Source_of_Lead'].value_counts())
print(df_raw['Region'].value_counts())

# Converting Str to Int
categorical_cols = ['Marital_Status','Prior_Insurance', 'Claims_Severity', 'Policy_Type', 'Source_of_Lead', 'Region']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_raw[col] = le.fit_transform(df_raw[col])

print(f'Encoded columns: {categorical_cols}')


# Feature Engineering Model
X = df_raw.drop(['Conversion_Status', 'Time_to_Conversion'], axis=1)
y = df_raw['Conversion_Status']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)

# List of Feature names
feature_list = list(X.columns)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
lr = LogisticRegression(max_iter=500)

# Recursive feature elimination
rfe = RFE(estimator=lr, n_features_to_select=6)
rfe.fit(X_train_scaled, y_train)

# Top 6 Recursive Features
selected_features = X_train.columns[rfe.support_]

# Bar plot of selected features
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), [1]*len(selected_features), color='skyblue')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Selected (1 = Yes)')
plt.title('RFE Selected Features')
plt.tight_layout()
#plt.savefig("rfe_selected_features.png", dpi=300, bbox_inches='tight')
plt.show()


# Accuracy of the model with RFE selected features
X_train_selected_scaled = X_train_scaled[:, rfe.support_]
X_test_selected_scaled = X_test_scaled[:, rfe.support_]

lr.fit(X_train_selected_scaled, y_train) 
rfe_y_pred = lr.predict(X_test_selected_scaled)
rfr_accuracy = accuracy_score(y_test, rfe_y_pred)
print('\nRecursive feature accuracy score (on test):')
print(rfr_accuracy)
print('\n')

# Running Random Forest Classifier ,Decision Tree Classifier, and Adaptive Boost Classifier off of RFE selected features
X_selected_train = X_train_scaled[:, rfe.support_]
X_selected_test = X_test_scaled[:, rfe.support_]

# Finding best classifier
rf_selected_classifier = {
    'Random Forest Classifier':RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Decision Tree Classifier':DecisionTreeClassifier(random_state=42),
    'Ada Boost Classifier':AdaBoostClassifier(random_state=42) 
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
    print(f'\nROC-AUC {name}: {roc_auc: .4f}')
    
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'\nConfusion Matrix{name}:')
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
    if name == 'Random Forest Classifier':
        importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances.head(10))
        plt.title('Top 3 Feature Importances - Random Forest (after RFE)')
        plt.xlabel('Importance')
        plt.tight_layout()
        #plt.savefig('feature_importance_auto_insurance.png', dpi=300, bbox_inches='tight')
        plt.show()

