import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data
dataset=pd.read_csv("suv_data.csv")


#Collecting X and Y
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

#splitting data into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#Scaling the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#importing logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train , y_train)

#predicting the value
y_pred = classifier.predict(x_test)

#calculating accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)
