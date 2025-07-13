#!/usr/bin/env python
# coding: utf-8

# In[168]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
os.chdir('C:\\Users\\asus\\OneDrive\\Desktop\\Final Portfolio Projects\\Telco Customer Churn Prediction')
df = pd.read_csv('Telco Customer Dataset.csv')

df.head()
df_copy = df.copy()


# In[169]:


df.shape


# In[170]:


df.describe()


# In[171]:


df.info()


# In[172]:


df.isna().sum()


# In[173]:


num_cols = df.select_dtypes(include = 'int64' or 'float64').columns.tolist()
cat_cols = df.select_dtypes(include = 'object').columns.tolist()


# In[174]:


df[num_cols].skew()


# In[175]:


df['Churn'].value_counts(normalize = True)


# In[176]:


plt.figure(figsize = (20, 10))
sns.countplot(x = df['Churn'])
plt.show()


# In[177]:


for cols in num_cols:
    plt.figure(figsize = (20, 10))
    sns.boxplot(df[cols])
    plt.title(f"Distribution Analysis of {cols}")
    plt.tight_layout()
    plt.show()


# In[178]:


df[num_cols].skew()


# In[179]:


num_cols.remove('SeniorCitizen') #Senior Citizen is binary - Not many Useful insights!


# In[180]:


num_cols


# In[181]:


cat_cols


# In[182]:


df = df.drop('customerID', axis = 1)
cat_cols.remove('customerID')


# In[183]:


df


# In[184]:


cat_cols


# In[185]:


df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})


# In[186]:


df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})


# In[187]:


df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})


# In[188]:


df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})


# In[189]:


df['OnlineSecurity'].value_counts()


# In[190]:


df['OnlineBackup'].value_counts()


# In[191]:


df['DeviceProtection'].value_counts()


# In[192]:


df['Contract'].value_counts()


# In[193]:


df['Contract'] = df['Contract'].replace({'Month-to-month': 'Monthly', 'Two year': 'BiYearly', 'One year': 'Yearly'})


# In[194]:


df['PaperlessBilling'] = df['PaperlessBilling'].replace({'No': 0, 'Yes': 1})


# In[195]:


df['PaymentMethod'].value_counts()


# In[196]:


df['TotalCharges'].head()


# In[197]:


df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})


# In[198]:


df


# In[199]:


df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 0, 'Yes': 1})


# In[200]:


df['InternetService'].value_counts()


# In[201]:


df['HasInternetService'] = df['InternetService'].apply(lambda x: 0 if x == 'No' else 1)


# In[202]:


cols_to_fix = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for cols in cols_to_fix:
    df[cols] = df[cols].map({'No internet service': 0, 'No': 0, 'Yes': 1})


# In[203]:


df['InternetService'].value_counts()


# In[204]:


df


# In[205]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')


# In[206]:


df['TotalCharges'].isna().sum()


# In[207]:


df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# In[208]:


df = pd.get_dummies(df, columns=['InternetService'], drop_first=False)


# In[209]:


df


# In[210]:


df = pd.get_dummies(df, columns = ['Contract'], drop_first = True)


# In[211]:


df['Automatic_Transfer'] = df['PaymentMethod'].apply(lambda x: 1 if x in ['Bank transfer (automatic)', 'Credit card (automatic)'] else 0)


# In[212]:


df = pd.get_dummies(df, columns = ['PaymentMethod'], drop_first = True)


# In[213]:


bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)


# In[214]:


df.head()


# In[215]:


df['Churn'].value_counts(normalize = True)


# In[216]:


plt.figure(figsize = (20, 10))
sns.countplot(x = df['Churn'])
plt.legend()
plt.tight_layout()
plt.show()


# In[217]:


bools_cols = df.select_dtypes(include = 'int64').columns.tolist()
bools_cols.remove('tenure')
bools_cols.remove('Churn')


# In[218]:


for cols in bools_cols:
    print(df[cols].mean())
    print(df.groupby(cols)['Churn'].mean())
    print('\n')


# In[219]:


for cols in bools_cols:
    plt.figure(figsize = (20, 10))
    sns.barplot(x = cols, y = 'Churn', data = df)
    plt.title(f"Churn Rate by {cols}")
    plt.tight_layout()
    plt.show()


# In[220]:


for cols in bools_cols:
    (pd.crosstab(df[cols], df['Churn'], normalize = True) * 100).plot(kind = 'bar', stacked = 'True', figsize = (20, 10))
    plt.title(f"CrossTabs {cols} vs. Churn")
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# In[221]:


from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list(
    name="orange_blue", 
    colors=["#FF7F00", "#FFFFBF", "#007FFF"]  # Orange, white, blue
)
corr = df.corr()
plt.figure(figsize = (20, 10))
sns.heatmap(corr, annot = True, cmap = custom_cmap)
plt.tight_layout()
plt.show()


# In[222]:


pd.pivot_table(df, values='Churn', index='Contract_Monthly', columns= 'InternetService_Fiber optic', aggfunc='mean')


# In[223]:


'''
#Row Contract_Monthly=0 (Non-Monthly Contracts):

For customers without Fiber Optic (0), the churn rate is 0.0359 (~3.6%).

For customers with Fiber Optic (1), the churn rate increases to 0.1395 (~13.9%).

#Row Contract_Monthly=1 (Monthly Contracts):

For customers without Fiber Optic (0), the churn rate is 0.2822 (~28.2%).

For customers with Fiber Optic (1), the churn rate is even higher at 0.5461 (~54.6%).

'''


# In[224]:


pd.pivot_table(df, values='Churn', index='SeniorCitizen', columns= 'PaymentMethod_Electronic check', aggfunc='mean')


# In[225]:


'''
For non-senior citizens (SeniorCitizen=0):

Customers not using electronic checks (PaymentMethod_Electronic check=0) have a churn rate of 0.1547 (~15.5%).

Customers using electronic checks (PaymentMethod_Electronic check=1) show a much higher churn rate of 0.4257 (~42.6%).

For senior citizens (SeniorCitizen=1):

Customers not using electronic checks (PaymentMethod_Electronic check=0) have a churn rate of 0.2901 (~29%).

Customers using electronic checks (PaymentMethod_Electronic check=1) show the highest churn rate at 0.5337 (~53.4%).

'''


# In[226]:


plt.figure(figsize = (20, 10))
sns.histplot(df['TotalCharges'], kde=True)
plt.title("Distribution of Total Charges")
plt.show()

plt.figure(figsize = (20, 10))
sns.boxplot(x = df['tenure'], y = df['TotalCharges'])
plt.title("Distribution: Tenure vs. TotalCharges")
plt.show()


# In[227]:


df['TotalCharges'].skew()
df['tenure'].skew()


# In[228]:


plt.figure(figsize = (20, 10))
sns.boxplot(y = df['TotalCharges'], x = df['Churn'])
plt.tight_layout()
plt.show()


# In[229]:


'''
Churning Customers (on Average) don't pay that much, it's those who don't pay higher charges tend to stay with us.
It's possible that the churn customers don't necessarily believe in our products and don't really want to continue with us because of that.
'''


# In[230]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#Trying Logistic Regression

df['TotalCharges'] = df['TotalCharges'].clip(lower = df['TotalCharges'].quantile(0.01), upper = df['TotalCharges'].quantile(0.99))
df['TotalCharges'] = np.log1p(df['TotalCharges'])

scaler = StandardScaler()
df['TotalCharges_scaled'] = scaler.fit_transform(df[['TotalCharges']])


# In[231]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

x_lr = df.drop(['Churn', 'TotalCharges'], axis = 1)
y_lr = df['Churn']
x = df.drop('Churn', axis = 1)
y = df['Churn']

x_temp_lr, x_test_lr, y_temp_lr, y_test_lr = train_test_split(x_lr, y_lr, random_state = 42, test_size = 0.25)
x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(x_temp_lr, y_temp_lr, random_state = 42, test_size = 0.2)

smote_lr = SMOTE(random_state = 42)
x_train_res_lr, y_train_res_lr = smote_lr.fit_resample(x_train_lr, y_train_lr)

x_temp, x_test, y_temp, y_test = train_test_split(x, y, random_state = 42, test_size = 0.25)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, random_state = 42, test_size = 0.2)

smote = SMOTE(random_state = 42)
x_train_resampled_temp, y_train_resampled_temp = smote.fit_resample(x_train, y_train)


# In[232]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

baseline_models = {
    'Random Forest': RandomForestClassifier(max_depth = 20, n_estimators = 20),
    'Decision Trees': DecisionTreeClassifier(max_depth = 20),
    'XGB': XGBClassifier(max_depth = 20),
    'CatBoost': CatBoostClassifier(),
    'Adaboost': AdaBoostClassifier() 
}

for name, model in baseline_models.items():
    model.fit(x_train_resampled_temp, y_train_resampled_temp)
    y_pred_temps = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred_temps)
    cnfm = confusion_matrix(y_train, y_pred_temps)
    clsfs = classification_report(y_train, y_pred_temps)

    print("Name: ", model)
    print("Accuracy Score: ", accuracy)
    print("Confusion Score: ", cnfm)
    print("Classification Score: ", clsfs)


# In[233]:


lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(x_train_res_lr, y_train_res_lr)
y_pred_lr = lr_model.predict(x_train_lr)

accuracy_lr = accuracy_score(y_train_lr, y_pred_lr)
cnfm_lr = confusion_matrix(y_train_lr, y_pred_lr)
clsfs_lr = classification_report(y_train_lr, y_pred_lr)

print('Accuracy Score: ', accuracy_lr)
print('Confusion Matrix: ', cnfm_lr)
print('Classification Report: ', clsfs_lr)


# In[234]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define model
model = LogisticRegression(solver='liblinear', class_weight='balanced')

# Define hyperparameters
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search
grid = GridSearchCV(model, param_grid, cv=skf, scoring='recall')  # Maximize recall
grid.fit(x_train_lr, y_train_lr)

print("Best Params:", grid.best_params_)
print("Best Recall:", grid.best_score_)


# In[247]:


from sklearn.model_selection import cross_val_predict
y_cv_proba = cross_val_predict(grid.best_estimator_, x_train, y_train, cv=skf, method='predict_proba')[:, 1]


# In[249]:


from sklearn.model_selection import GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}
param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'scale_pos_weight': [1, 5, 10]
}

grid_rf = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid_rf, scoring = 'f1', n_jobs = -1, verbose = 0, cv = 5)
grid_dt = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = param_grid_dt, scoring = 'f1', n_jobs = -1, verbose = 0, cv = 5)
grid_xgb = GridSearchCV(estimator = XGBClassifier(), param_grid = param_grid_xgb, scoring='f1', n_jobs = -1, verbose = 0, cv=5)

grid_rf.fit(x_train, y_train)
grid_dt.fit(x_train, y_train)
grid_xgb.fit(x_train, y_train)

best_rf = grid_rf.best_estimator_
best_df = grid_dt.best_estimator_
best_xgb = grid_xgb.best_estimator_


# In[251]:


print(grid_rf.best_params_)
print(grid_dt.best_params_)
print(grid_xgb.best_params_)


# In[253]:


print(best_rf)
print(best_df)
print(best_xgb)


# In[255]:


print(grid_rf.best_score_)
print(grid_dt.best_score_)
print(grid_xgb.best_score_)


# In[257]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score
from sklearn.base import clone

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

f1s = []
accuracy = []
recall = []

for train_idx, val_idx in skf.split(x_train, y_train):
    x_fold_train, y_fold_train = x_train.iloc[train_idx], y_train.iloc[train_idx]
    x_fold_val, y_fold_val = x_train.iloc[val_idx], y_train.iloc[val_idx]
    smote = SMOTE(random_state = 42)
    x_resampled, y_resampled = smote.fit_resample(x_fold_train, y_fold_train)

    model = clone(best_xgb)
    model.fit(x_resampled, y_resampled)

    preds = model.predict(x_fold_val)
    f1_s = f1_score(y_fold_val, preds)
    accus = accuracy_score(y_fold_val, preds)
    recs = recall_score(y_fold_val, preds)
    f1s.append(f1_s)
    accuracy.append(accus)
    recall.append(recs)

print("Average F-1 Score: ", np.mean(f1s))
print("Average Accuracy Score: ", np.mean(accuracy))
print("Average Recall: ", np.mean(recall))
print("Accuracy scores for each fold:", accuracy)
print("Recall scores for each fold:", recall)
print("F-1 Score for each fold: ", f1s)


# In[259]:


y_proba = model.predict_proba(x_train)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f1s = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    score = f1_score(y_train, y_pred)
    f1s.append(score)

best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
best_f1 = f1s[best_idx]

print(f'Best threshold: {best_threshold:.2f}')
print(f'Best F1 Score: {best_f1:.4f}')


# In[261]:


plt.figure(figsize = (20, 10))
plt.plot(thresholds, f1s)
plt.xlabel("Threshold")
plt.ylabel("F1-Scores")
plt.title("Threshold vs F-1 Scores")
plt.tight_layout()
plt.show()


# In[263]:


print("Threshold = 0.5")
print(classification_report(y_train, (y_proba >= 0.5).astype(int)))

print("Threshold = 0.7")
print(classification_report(y_train, (y_proba >= 0.7).astype(int)))


# In[265]:


model = best_xgb
y_cv_proba = cross_val_predict(model, x, y, cv=5, method = 'predict_proba')[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = []

for t in thresholds:
    y_pred = (y_cv_proba >= t).astype(int)
    f1s.append(f1_score(y, y_pred))
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
best_f1 = f1s[best_idx]

plt.figure(figsize = (20, 10))
plt.plot(thresholds, f1s, marker = 'o')
plt.axvline(best_threshold, color = 'red', linestyle = '--', label = f'Best Threshold = {best_threshold: .2f}')
plt.title("F-1 Score vs. Threshold (Cross-Val)")
plt.xlabel("Threshold")
plt.ylabel("F-1 Score")
plt.grid(True)
plt.legend()
plt.show()

print("Best Threshold: ", best_threshold)
print("Best F1", best_f1)
y_pred = (model.predict_proba(x)[:, 1] >= 0.55).astype(int)


# In[267]:


y_cv_pred = (y_cv_proba >= best_threshold).astype(int)
print(classification_report(y, y_cv_pred))


# In[327]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
import numpy as np

# Define your feature and label sets
X = x_train_lr.reset_index(drop=True)
Y = y_train_lr.reset_index(drop=True)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)
model = LogisticRegression(C=10, penalty='l1', solver='liblinear')

probas = np.zeros(len(Y))  # This will store the OOF probs

for train_idx, val_idx in skf.split(X, Y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    # Apply SMOTE to training folds only
    X_res, Y_res = smote.fit_resample(X_train, Y_train)

    model.fit(X_res, Y_res)

    # Predict on the current fold's val set
    probas[val_idx] = model.predict_proba(X_val)[:, 1]  # Store correctly

# Now probas[] holds CV-based predicted probs for all training samples

# Tune threshold for best recall
thresholds = np.arange(0.3, 0.9, 0.01)
f1s = [f1_score(Y, (probas >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(recalls)]
best_f1s = max(f1s)

print(f"✅ Best Threshold: {best_threshold:.2f}")
print(f"🎯 Best F-1 Score: {best_f1s:.4f}")


# In[291]:


lr_model = LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear')
lr_model.fit(x_train_res_lr, y_train_res_lr)

y_proba_lr_val = lr_model.predict_proba(x_val_lr)[:, 1]
y_preds_lr_val = (y_proba_lr_val >= 0.3).astype(int)

print("Accuracy:", accuracy_score(y_val_lr, y_preds_lr_val))
print("Confusion Matrix:\n", confusion_matrix(y_val_lr, y_preds_lr_val))
print("Classification Report:\n", classification_report(y_val_lr, y_preds_lr_val))


# In[329]:


lr_model = LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear', class_weight={0: 1, 1: 1.5})
y_cv_proba = cross_val_predict(lr_model, x, y, cv=10, method='predict_proba')[:, 1]
lr_model.fit(x_train_res_lr, y_train_res_lr)

y_proba_val = lr_model.predict_proba(x_val_lr)[:, 1]
y_pred_lr_val = (y_proba_val >= best_threshold).astype(int)

accu = accuracy_score(y_val_lr, y_pred_lr_val)
cnfm = confusion_matrix(y_val_lr, y_pred_lr_val)
cr = classification_report(y_val_lr, y_pred_lr_val)

print("Accuracy Score: ", accu)
print("Confusion Matrix: ", cnfm)
print("Classification Report", cr)


# In[323]:


from sklearn.metrics import precision_score
threshold = np.arange(0.1, 0.9, 0.1)
pr = []
rec = []
f1 = [f1_score(Y, (probas >= t).astype(int)) for t in thresholds]

for t in thresholds:
    preds = (probas >= t).astype(int)
    pr.append(precision_score(Y, preds))
    rec.append(recall_score(Y, preds))

plt.figure(figsize = (20, 10))
plt.plot(thresholds, pr, marker = 'o', label = 'Precision')
plt.plot(thresholds, rec, marker = 'x', label = 'Recall')
plt.axvline(best_threshold, color = 'orange', linestyle = '--', label = f'Best Threshold = {best_threshold}')
plt.title("Precision vs. Recall vs. Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.show()


# In[331]:


plt.savefig("threshold_tuning_plot.png")


# In[333]:


lr_model = LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear', class_weight={0: 1, 1: 1.5})

lr_model.fit(x_train_res_lr, y_train_res_lr)
y_pred_final = lr_model.predict(x_test_lr)

accu = accuracy_score(y_test_lr, y_pred_final)
clsr = classification_report(y_test_lr, y_pred_final)
cnfm = confusion_matrix(y_test_lr, y_pred_final)

print("✅Accuracy on Final Test Set: ", accu)
print("✅Confusion Matrix of Final Test Set: ", cnfm)
print("✅Classification Report on Final Set: ", clsr)


# In[ ]:




