# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Prabhakaran P
RegisterNumber:  212224040236
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
##   Data Head
<img width="1115" alt="318635076-bc753f9a-8a09-4815-89ae-79ec8e165e8a" src="https://github.com/user-attachments/assets/858372a5-ba41-419d-8ef4-fed4b06fc0da" />

## DATASET INFO
<img width="1115" alt="318635303-f760819f-e78e-4a12-8b8c-32daeacb3410" src="https://github.com/user-attachments/assets/8c52cf3f-2248-497c-a8a9-c74272564c14" />

## NULL DATASET
<img width="1115" alt="318635394-67c8a973-c928-44e6-be58-ffd98b45057b" src="https://github.com/user-attachments/assets/df862cae-38d5-4e8d-9dd2-455341f2ce17" />

## VALUES COUNT IN LEFT COLUMN
<img width="1115" alt="318635551-01f07495-4958-4186-bee8-c6632195895e" src="https://github.com/user-attachments/assets/de4a3b27-7418-41e5-9a62-c9bd1fa3981a" />

## DATASET TRANSFORMED HEAD
<img width="1115" alt="318635713-1138eafe-ce6f-4fde-85e9-71daf0b64c00" src="https://github.com/user-attachments/assets/6d2efb8e-d3de-4faf-9fca-126edd10c01b" />

## X.HEAD
<img width="1115" alt="318635986-569580d7-a8b0-4d20-a637-73bf2f4bc9fe" src="https://github.com/user-attachments/assets/05b18c2a-e45a-4576-8921-f9803d570eaa" />

## ACCURACY
<img width="1115" alt="318635869-2dfb7e6d-2f2c-4b47-a09f-f766553e9dc5" src="https://github.com/user-attachments/assets/c976d69a-5d1c-44e0-9977-b3b74c519a5b" />

## DATA PREDICTION
<img width="1115" alt="318636527-0e11798e-c2ab-44ee-875b-982d940a1851" src="https://github.com/user-attachments/assets/e8e30eee-b36e-4417-8ff4-d05149d12e3f" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
