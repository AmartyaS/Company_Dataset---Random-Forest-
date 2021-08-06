# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 01:02:08 2021

@author: ASUS
"""


#Importing all the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Loading the Data File
file=pd.read_csv("D:\Data Science Assignments\Python-Assignment\Random Forest\Company_Data.csv")
file.describe()

#Data Manipulation
data=file
data.describe()
data.columns
data.dtypes
data.Sales
data.isna().sum()
data['Sales']=pd.cut(file['Sales'], bins=(-1,4,8,12,17), labels=["Bad","Medium","Good","Best"])
data['Urban']=[1 if x=='Yes' else 0 for x in file['Urban']]
data['US']=[1 if x=='Yes' else 0 for x in file['US']]
data=pd.get_dummies(data,columns=["ShelveLoc"])

#Initialising the target and the predictor variables
x=data.iloc[:,1:13]
y=pd.DataFrame(data.iloc[:,0])

#Splitting the dataframe into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#Making the model with 100 number of trees
model1=RandomForestClassifier(n_estimators=100)
model1.fit(x_train,y_train)
preds1=model1.predict(x_test)
#Checking the accuracy of the model
print("Accuracy : ",metrics.accuracy_score(y_test,preds1))


#Making the model with 10 number of trees
model2=RandomForestClassifier(n_estimators=10)
model2.fit(x_train,y_train)
preds2=model2.predict(x_test)
#Checking the accuracy of the model
print("Accuracy : ",metrics.accuracy_score(y_test,preds2))


#Making a model with collection of n_estimators to check the trend of its accuracy
a=[]
for i in range(1,101,4):
    model=RandomForestClassifier(n_estimators=i)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=metrics.accuracy_score(y_test,y_pred)
    a.append([i,acc])
accuracy=pd.DataFrame(a)
accuracy.columns=["N_Estimators","Accuracy_Values"]


#Visualizing the trend of accuracy of the model with increasing N_Estimator
plt.plot(accuracy.N_Estimators,accuracy.Accuracy_Values)
plt.title("Accuracy Values Vs N_Estimators")
plt.xlabel("N_Estimators")
plt.ylabel("Accuracy Values")
plt.show()

#Marking the important features of the dataframe
feature_imp=pd.Series(data=model.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_imp
#Visualizing the important features of the dataframe
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.title("Visualizing Important Feature")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()
