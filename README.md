# People-Classification

# 1 Introduction
## 1.1 Aim of Project
In this project, a dataset containing ten different features are given. We are expected to build and train our model using the training and validating set. Then the class label in the testing set should be predicted using the model we built.
## 1.2 Dataset
Data instance consists of the following input features and the target class label.

<img width="661" alt="image" src="https://user-images.githubusercontent.com/71811311/235942778-b9e28060-65d4-4953-8375-ab4f571f0b80.png">

Train.csv and Validate.csv files provide the target class labels as ground truth (string).For almost all the attributes, there is a possible value “nan” which means “unavailable”. Some data- processing strategies can be used. The Train.csv and Validate.csv contain the information of users. Test.csv contains the information of the users without the target class label. 
## 1.3 Objectives
(1) To apply the data processing techniques presented in lectures to clean the data set.
(2) To identify important features and effectively incorporate the features into the model. 
(3) To decide on the appropriate type of classifier to use for the given the dataset, as well as optimize the classifier’s parameters to achieve the best possible performance. 
(4) To evaluate the performance of the model, using the metric of accuracy.

# 2 Data Visualization and Processing
## 2.1 Data Visualization
The dataset given contains ten different features and the last column (Class) is the target we need to predict in the test set. 

 ![image](https://user-images.githubusercontent.com/71811311/235941035-daedf0eb-ba98-40f9-8616-e73b4f582262.png)

The data types include numerical value and categorial value. The first step is to convert the categorial variables in to numerical values using the map() function in Python. Below is the result after conversion.

 ![image](https://user-images.githubusercontent.com/71811311/235941092-3bfc729e-91d7-48da-9deb-4912edb17b82.png)

## 2.2 Missing Values
Most of the features contain missing values, so we need to choose a method to fill in the missing values for better analysis. The techniques we used for different algorithms vary and it will be discussed in details in the following sections.

 ![image](https://user-images.githubusercontent.com/71811311/235941121-221651a4-e8ca-41c9-8bfe-628e08d78ea6.png)


# 3 Algorithms and Implementation
## 3.1 Random Forest
### 3.1.1 Missing Values
For this model, Multiple Imputation by Chained Equations (MICE) is used to fill in the missing values. The idea of multiple imputation (MI) comes from Bayesian estimation, which assumes that the value to be imputed is random and its value comes from the observed value. In practice, the values to be interpolated are usually estimated, and then various noise values are added to form multiple sets of optional interpolated values. Use a selection basis to choose the most appropriate interpolation value.
The methods for handling missing values, such as matching imputation and mean substitution, are methods of simple imputation, and multiple imputation compensates for the shortcomings of simple imputation. It does not attempt to estimate each missing value by simulated values but is a random sample of missing data values (these samples can be combinations of different model fit results). The implementation of such procedures adequately reflects the uncertainty due to missing values so that the statistical conclusions are valid.
The dataset is imputed using the miceforest Python library, which uses Random Forest. The dataset after imputation is shown below.

 ![image](https://user-images.githubusercontent.com/71811311/235941187-bcd05361-43ec-4a03-9dea-945091618b70.png)

### 3.1.2 Feature Selection
First, column ID is removed since it is not relevant to the prediction result. A correlation matrix is plotted to analyze the relation between each feature.

 ![image](https://user-images.githubusercontent.com/71811311/235941245-80c04465-aa29-4159-afd4-7fe1c75ea987.png)

This figure shows that Gender has a strong negative correlation with Spending_Score. Age has a positive correlation with Spending_Score and Class. Profession has a negative correlation with Age and Graduate. Therefore, we can try to remove one of the features that are highly correlated with others. The feature_selection function in sklearn library is used to find the set of features that gives the best result. We select different number of features to build the model and compare the accuracy. 

 ![image](https://user-images.githubusercontent.com/71811311/235941296-3c908aef-9dc5-4611-8546-4baecc896ff6.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941317-74c8b122-7dad-4158-80cb-58404e3955cc.png)

 
The best result is obtained when all features are included. 
### 3.1.3 Hyper-parameter Adjusting
To achieve better results, the hyper-parameters max_features and n_estimators are adjusted in the model. The max_features is the maximum number of features per decision tree. Since there are nine features, this hyper-parameter is looped from 1 to 9. The n_estimators is the number of trees in the forest, and it is looped from 1 to 100 to find the best result. The accuracy is calculated by 10-fold cross validation.

 ![image](https://user-images.githubusercontent.com/71811311/235941436-0067c3a8-1f00-4b54-877e-74a27d7c463d.png)

For each max_features and n_estimators pair, accuracy is calculated with different random state for 10 times and the result shows that max_features = 3, n_estimators = 97 gives the most stable accuracy among all pairs. Therefore, the model with these two values is used for predicting the target in the test set. 
3.1.4 Final Model and Result
To summarize, the Random Forest model is built with all features and the hyper-parameter is set to max_features = 3, n_estimators = 97. This is the final model and the accuracy on the validate set. The accuracy is around 0.425.

 ![image](https://user-images.githubusercontent.com/71811311/235941468-5d404792-3a0f-480c-a248-b2b1db49fd0a.png)

Below is the confusion matrix of Random Forest.
 
 ![image](https://user-images.githubusercontent.com/71811311/235941498-e45b83eb-bf94-45cc-a96a-28f0c00677b6.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941537-a97ac2a3-2724-468c-a184-d6169cf5fd58.png)

Below is the result of prediction.
 
![image](https://user-images.githubusercontent.com/71811311/235941563-8939567d-8838-4f4c-849e-b286ec2920c3.png)

## 3.2 Multilayer Perceptron
### 3.1.1 Data Processing
First, we use LabelEncoder to transform String type feature into int64 type for model training. Then, for the missing feature values, we use random fill for feature Gender, Graduate, Profession, and Category and use mean value fill for feature Years_of_Working and Family_Members. Finally, for input feature, we use Z-Score Normalization. And for label, we use one-hot encoding.

 ![image](https://user-images.githubusercontent.com/71811311/235941609-07ebcccf-0425-4d45-bb3c-2a1f11e35a4f.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941636-040142ce-7dcc-4167-b2eb-8ea811b007a5.png)

![image](https://user-images.githubusercontent.com/71811311/235941655-f2ec77ef-fa73-49e4-ae6e-6bb113c50f62.png)

 
 
### 3.1.2 Feature Selection
For feature selection, we use SelectKBest and f_classif to analyse the feature score, the result show that the Top 6 score features are Gender, Age, Graduate, Profession, Spending_Score and Family_Members. So, we will use these six features for model training.

 ![image](https://user-images.githubusercontent.com/71811311/235941692-8bfa462a-0f71-4361-924e-c1569f9c311e.png)

### 3.1.3 Network configuration
In this part, we use 4 layers perceptron with 2 hidden layers, first hidden layer has 200 hidden nodes, second hidden layer has 100 hidden nodes. Both use sigmoid activation function. We use six feature Gender, Age, Graduate, Profession, Spending_Score and Family_Members as input. Batch size is 20 and epochs is 100. The structure is shown as below.

 ![image](https://user-images.githubusercontent.com/71811311/235941721-906688b7-97ce-4158-920f-b9f2c0220ade.png)


### 3.1.4 Result and analysis
The training result is shown as below, from which we have several observations. First, we can see that the validation accuracy is 0.4733. Second, the learning curve illustrates that the model does not have overfitting. Third, from confusion matrix we can see that class B is the class with the lowest prediction accuracy.

 ![image](https://user-images.githubusercontent.com/71811311/235941752-a3bd4579-3b84-4300-ace7-51de94af1d59.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941773-205c9575-c86a-44bd-8049-bd62f081ac38.png)

![image](https://user-images.githubusercontent.com/71811311/235941787-043b5dad-417e-41c9-a092-73e6210d6bc0.png)

![image](https://user-images.githubusercontent.com/71811311/235941814-fe4dfbc7-24b3-401e-b63f-9a2ad0e1ede0.png)

 
 
 
## 3.3 Naïve Bayes
### 3.3.1 Missing Values
In this model, using the mode imputation to fill the missing values. Using the statistics Python library to count the most frequent values of each feature, and then filling the missing values by these most frequency values. The python code and the dataset after imputation are shown as follow.
 
 ![image](https://user-images.githubusercontent.com/71811311/235941846-2afd3533-e54c-4b49-9962-761588d93b41.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941870-fbf2dfdb-326a-4542-aee9-9d1aeb13d7bc.png)

### 3.3.2 Feature Selection
In this model, using the methods that based on statistical tests of univariate characteristics to select the features that can achieve best results. One of the functions in sklearn.feature_selection is SelectKBest, this function can find the k best features, this is based on chi-square test. Changing the k can get different accuracy.

 ![image](https://user-images.githubusercontent.com/71811311/235941898-c9d55584-c0a9-4c50-9a05-bd6fb62cb690.png)

The best result is achieved when 8 features included.
The picture below shows the features that be chose to build to final model.

 ![image](https://user-images.githubusercontent.com/71811311/235941922-938f060b-3ab6-48a0-a15b-af228b4222f7.png)

### 3.3.3 Building Model
Using the Naïve Bayes to build the model. In the sklearn module in python, there are three Naïve Bayes classification methods in total. Using the GaussianNB to build the Naïve Bayes model. The results of train dataset and validate dataset are shown as follow.

 ![image](https://user-images.githubusercontent.com/71811311/235941952-ab92c86b-345b-4d8d-be9f-093807604181.png)
 
![image](https://user-images.githubusercontent.com/71811311/235941972-139df776-9e41-4ff5-930f-daaff8db7e39.png)

 
Below is the confusion matrix.

 ![image](https://user-images.githubusercontent.com/71811311/235941999-670790c0-afc3-4054-bf38-cd04b8c0b6a0.png)

The prediction result of test dataset shows below.

 ![image](https://user-images.githubusercontent.com/71811311/235942018-32fd6b8f-d513-4a1d-b01b-23464db7c965.png)


## 3.4 K-Nearest Neighbors
### 3.4.1 Missing Values & Type change
For the missing values in the dataset, we use SimpleImputer of sklearn library to fill the empty values. There are two types of missing data: one is 1 or 0 data like gender and graduate, one is integer value like family members and working hours. For the first kind of data, we use “most frequent” way to fill them, and for the second, we use median number to fill in. Here’s the codes:

 ![image](https://user-images.githubusercontent.com/71811311/235942041-78e19e64-f5b0-4c41-94a4-c82276a6cced.png)

Then, we need to change the string types data to integer type for further research. We used get_dummies() function from pandas library. It will generate new features according to different types. Here’s the codes:

 ![image](https://user-images.githubusercontent.com/71811311/235942084-ab023c96-6a17-43b6-bba4-5ca7ea1f8e66.png)


### 3.4.2 Feature selection
In this method, we use PCA to do the feature selection. Then, we ran a test program to decide how many features we choose according to the PCA results of training set. So, we tested the accuracy for different n_components in validation set. Here’s the result graph for validation set:

 ![image](https://user-images.githubusercontent.com/71811311/235942118-a272deb3-e677-4ec2-9fa9-96af285a7f5e.png)

According to the graph, we chose n_components = 4 for PCA.
### 3.4.3 Building Model
The model we use for this method is K-Nearest Neighbours, which is also called KNN. It is a very basic method in Machine Learning. The k-nearest neighbours algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. We can just use sklearn package to implement it.

 ![image](https://user-images.githubusercontent.com/71811311/235942152-5bb300ea-5366-4be6-97d8-823fb925af47.png)
 
![image](https://user-images.githubusercontent.com/71811311/235942172-d7d05b4b-44c0-4792-adce-f9d1c40be8f4.png)

 
For the validation set, the whole process is generally the same as training set. Here’s the validation set score and its confusion matrix:

 ![image](https://user-images.githubusercontent.com/71811311/235942191-570bc829-d6dd-4e58-a017-5e655c862eb6.png)
 
![image](https://user-images.githubusercontent.com/71811311/235942212-c0b82c70-e8d0-4b4d-9d7b-aabeb25c8a5b.png)

 

# 4 Results and Discussion

<img width="639" alt="image" src="https://user-images.githubusercontent.com/71811311/235942530-030d706e-b07c-4bc6-bacd-0d2d70433874.png">


Four algorithms are used to design the classifier to predict the target class label of different users in this project and the table above shows the results. A comparison of the results in the table shows the KNN algorithm achieved the best results in train dataset. But it got the lowest accuracy in validate dataset. Overall, the Multilayer Perception algorithm achieved the best result. Therefore, we chose the Multilayer Perception to be the algorithm for final prediction of the test data. 






