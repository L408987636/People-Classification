#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
import miceforest as mf


# In[169]:


# read data
data = pd.read_csv('D:\HKU-Subject Material\SEM2\COMP7103\Assignment2\dataset\Train.csv')
data_test = pd.read_csv('D:\HKU-Subject Material\SEM2\COMP7103\Assignment2\dataset\Test.csv')
data_val = pd.read_csv('D:\HKU-Subject Material\SEM2\COMP7103\Assignment2\dataset\Validate.csv')
data_test = data_test.drop(['Class(Target)'],axis=1)


# In[170]:


data


# In[171]:


data.isnull().sum()


# In[172]:


col_missing = data.columns[data.isnull().any()].values


# In[173]:


# Map categorial value to numerical
data['Spending_Score'] = data['Spending_Score'].map({'High':3.0, 'Average':2.0, 'Low':1.0})
data['Profession'] = data['Profession'].map({'Artist':0, 'Homemaker':1, 'Lawyer':2, 'Entertainment':3, 'Doctor':4,
       'Engineer':5, 'Healthcare':6, 'Executive':7, 'Marketing':8})
data['Category'] = data['Category'].map({'Cat_1':1, 'Cat_2':2, 'Cat_3':3, 'Cat_4':4, 'Cat_5':5, 'Cat_6':6,'Cat_7':7})
data_val['Spending_Score'] = data_val['Spending_Score'].map({'High':3.0, 'Average':2.0, 'Low':1.0})
data_val['Profession'] = data_val['Profession'].map({'Artist':0, 'Homemaker':1, 'Lawyer':2, 'Entertainment':3, 'Doctor':4,
       'Engineer':5, 'Healthcare':6, 'Executive':7, 'Marketing':8})
data_val['Category'] = data_val['Category'].map({'Cat_1':1, 'Cat_2':2, 'Cat_3':3, 'Cat_4':4, 'Cat_5':5, 'Cat_6':6,'Cat_7':7})


# In[174]:


data_test['Spending_Score'] = data_test['Spending_Score'].map({'High':3.0, 'Average':2.0, 'Low':1.0})
data_test['Profession'] = data_test['Profession'].map({'Artist':0, 'Homemaker':1, 'Lawyer':2, 'Entertainment':3, 'Doctor':4,
       'Engineer':5, 'Healthcare':6, 'Executive':7, 'Marketing':8})
data_test['Category'] = data_test['Category'].map({'Cat_1':1, 'Cat_2':2, 'Cat_3':3, 'Cat_4':4, 'Cat_5':5, 'Cat_6':6,'Cat_7':7})


# In[175]:


data


# In[176]:


stat_df = pd.DataFrame({'# of miss':data.drop(['ID','Class(Target)'], axis=1).isnull().sum(),
                        '% of miss':data.drop(['ID','Class(Target)'], axis=1).isnull().sum() / len(data) * 100,
                        'var':data.drop(['ID','Class(Target)'], axis=1).var()})
stat_df


# In[177]:


# Fill in missing values in train set using MICE
# Create kernel.
kds = mf.ImputationKernel(
  data.iloc[:,1:10],
  save_all_iterations=True,
  random_state=42
)

# Run the MICE algorithm for 30 iterations
kds.mice(30)

# Return the completed dataset.
data_complete = kds.complete_data()


# In[178]:


# Fill in missing values in validate set
# Create kernel. 
kds_val = mf.ImputationKernel(
  data_val.iloc[:,1:10],
  save_all_iterations=True,
  random_state=42
)

# Run the MICE algorithm for 30 iterations
kds_val.mice(30)

# Return the completed dataset.
data_val_cpl = kds_val.complete_data()


# In[179]:


# Fill in missing values in test set
# Create kernel. 
kds_val = mf.ImputationKernel(
  data_test.iloc[:,1:10],
  save_all_iterations=True,
  random_state=42
)

# Run the MICE algorithm for 30 iterations
kds_val.mice(30)

# Return the completed dataset.
data_test_cpl = kds_val.complete_data()


# In[180]:


# Convert class label into numbers
data_complete['Class'] = data['Class(Target)'].map({'A':1.0, 'B':2.0, 'C':3.0, 'D':4.0})
data_val_cpl['Class'] = data_val['Class(Target)'].map({'A':1.0, 'B':2.0, 'C':3.0, 'D':4.0})
data_val_cpl


# In[181]:


# Correlation matrix
plt.rcParams['figure.figsize'] = (20, 15) 
sns.heatmap(data_complete.corr(), annot = True, linewidths=.5, cmap="YlGnBu") 
plt.title('Correlation between features', fontsize = 30)
plt.tight_layout()


# In[182]:


X_train = data_complete.iloc[:,0:9].values
Y_train = data['Class(Target)'].values
X_val = data_val_cpl.iloc[:,0:9].values
Y_val = data_val['Class(Target)'].values
X_test = data_test_cpl.iloc[:,0:9].values


# In[183]:


# Standardize the variable
std=StandardScaler()
X_train=std.fit_transform(X_train)
X_val=std.transform(X_val)
X_test = std.transform(X_test)


# In[41]:


# feature selection, select 3 to 9 features to build the model and compare accuracy
for i in range(3,10):
    # train set
    X = np.array(data_complete.iloc[:,0:9].values)
    Y = np.array(data_complete['Class'].values)
    skb = SelectKBest(score_func=f_regression, k=i)
    skb.fit(X, Y.ravel())
    print('Features selected:', [data_complete.iloc[:,0:9].columns[j] for j in skb.get_support(indices = True)])
    X_select = skb.transform(X)
    Y_train = data['Class(Target)'].values
    
    # validate set
    Xv = np.array(data_val_cpl.iloc[:,0:9].values)
    Yv = np.array(data_val_cpl['Class'].values)
    skb = SelectKBest(score_func=f_regression, k=i)
    skb.fit(Xv, Yv.ravel())
    X_val_select = skb.transform(Xv)
    Y_val = data_val['Class(Target)'].values
    
    # build rf model
    rfc_best = RandomForestClassifier(max_features=3, n_estimators=84, random_state=42)
    rfc_best.fit(X_select,Y_train)
    rfc_pred = rfc_best.predict(X_val_select)
    rfc_acc =  cross_val_score(rfc_best, X_val_select, Y_val, cv=10, scoring = 'accuracy').mean()
    print('Feature num: ', i, 'Accuracy: ', rfc_acc)
    


# In[43]:


# Find the best hyper parameters
acc_list = [0]*X_train.shape[1]
for j in range(0, len(acc_list)):
    n_trees = []
    rfc_acc_best = 0
    #For cycle to choose the best n_estimators
    for i in range(100):
        rfc = RandomForestClassifier(max_features=j+1, n_estimators=i+1, random_state=42)
        rfc.fit(X_train,Y_train)
        rfc_pred = rfc.predict(X_val)
        rfc_acc = cross_val_score(rfc, X_val, Y_val, cv=10, scoring = 'accuracy').mean()
        #Make the best predicted value into RFC_PRED
        if rfc_acc > rfc_acc_best:
            RFC_PRED = rfc_pred
            rfc_acc_best = rfc_acc
        n_trees.append(rfc_acc)
    acc_list[j] = rfc_acc_best
    print('max_feature: ',j+1, 'The accuracy is :',rfc_acc_best,'The proper trees is :',1+n_trees.index(max(n_trees)))


# In[184]:


# Best parameters are used to build final model 
rfc_best = RandomForestClassifier(max_features=3, n_estimators=84, random_state=42)
rfc_best.fit(X_train,Y_train)
rfc_pred = rfc_best.predict(X_val)
rfc_train_acc =  cross_val_score(rfc_best, X_train, Y_train, cv=10, scoring = 'accuracy').mean()
rfc_acc =  cross_val_score(rfc_best, X_val, Y_val, cv=10, scoring = 'accuracy').mean()
print('Training accuracy of Random Forest: ', rfc_train_acc)
print('Validate accuracy of Random Forest: ', rfc_acc)


# In[185]:


# Compute confusion matrix
cm=confusion_matrix(Y_val, rfc_pred)

# Plot confusion matrix
ax = sns.heatmap(cm, annot=True, cmap='Blues',fmt='d')

ax.set_title('Confusion Matrix of Random Forest\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

print(metrics.classification_report(Y_val,y_pred=rfc_pred)) 


# In[112]:


# Prediction for test set
data_test = pd.read_csv('D:\HKU-Subject Material\SEM2\COMP7103\Assignment2\dataset\Test.csv')
data_test['Class(Target)']=pred_test
data_test


# In[113]:


data_test.to_csv('D:\HKU-Subject Material\SEM2\COMP7103\Assignment2\dataset\predict.csv')


# In[ ]:




