import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier


df = read_csv('Train.csv')
imr = SimpleImputer(missing_values=np.nan, strategy='median')
df['Years_of_Working '] = imr.fit_transform(df[['Years_of_Working ']])
df['Family_Members'] = imr.fit_transform(df[['Family_Members']])

imr2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['Gender'] = imr2.fit_transform(df[['Gender']])
df['Graduate'] = imr2.fit_transform(df[['Graduate']])

y = df.iloc[:,-1]
df = df.drop(['Class(Target)','ID'],axis=1)

pf = pd.get_dummies(df[['Category']])
df = pd.concat([df, pf], axis=1)
df.drop(['Category'], axis=1, inplace=True)

pf = pd.get_dummies(df[['Profession']])
df = pd.concat([df, pf], axis=1)
df.drop(['Profession'], axis=1, inplace=True)

pf = pd.get_dummies(df[['Spending_Score']])
df = pd.concat([df, pf], axis=1)
df.drop(['Spending_Score'], axis=1, inplace=True)
result = []
index = []

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents)

print(principalDf)
clf = KNeighborsClassifier()
clf.fit(principalDf, y)
print("Training set score:",clf.score(principalDf, y))


test = read_csv('Validate.csv')

imr = SimpleImputer(missing_values=np.nan, strategy='median')
test['Years_of_Working '] = imr.fit_transform(test[['Years_of_Working ']])
test['Family_Members'] = imr.fit_transform(test[['Family_Members']])

imr2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
test['Gender'] = imr2.fit_transform(test[['Gender']])
test['Graduate'] = imr2.fit_transform(test[['Graduate']])

y_test = test.iloc[:,-1]

test = test.drop(['Class(Target)','ID'],axis=1)

pf2 = pd.get_dummies(test[['Category']])
test = pd.concat([test, pf2], axis=1)
test.drop(['Category'], axis=1, inplace=True)

pf2 = pd.get_dummies(test[['Profession']])
test = pd.concat([test, pf2], axis=1)
test.drop(['Profession'], axis=1, inplace=True)

pf2 = pd.get_dummies(test[['Spending_Score']])
test = pd.concat([test, pf2], axis=1)
test.drop(['Spending_Score'], axis=1, inplace=True)

print(test)
principalComponents_test = pca.fit_transform(test)
principalDf_test = pd.DataFrame(data = principalComponents_test)

print("Validation set score:", clf.score(principalDf_test,y_test))

    # result.append(clf.score(principalDf_test,y_test))
    # index.append(i)
# #
y_pred = clf.predict(principalDf_test)

# test = read_csv('Test.csv')
# test['Class(Target)'] = y_pred
# test.to_csv("Predict_KNN.csv",index = None)


#
# cm = confusion_matrix(y_test, y_pred)
#
# ConfusionMatrixDisplay(cm).plot()
#
# plt.show()

#
#
#
