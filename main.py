import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#reading csv file from pandas
import pandas as pd
df = pd.read_csv(r'C:\Users\riapi\OneDrive\Documents\MachineLearning\WineQuality\winequality-red.csv')
print(df)

#data preprocessing
print(df.quality.unique())

#data visualization
#plt.bar(df['quality'], df['alcohol'], color='pink')
#plt.xlabel('Quality')
#plt.ylabel('Alcohol')

#Splitting the data
X = df.drop('quality', axis=1)
print(X)

Y = df['quality']
print(Y)

#splitiing into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Standardization
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
print(X_train)
print(X_test)

#DECISION TREE
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

y_pred_dtc = dtc.predict(X_test)
print(accuracy_score(Y_test, y_pred_dtc))
lr_acc = accuracy_score(Y_test, y_pred_dtc)

results = pd.DataFrame()
print(results)

tempresults = pd.DataFrame({'Algorithm': ['Decision Tree Method'], 'Accuracy': [lr_acc]})
results = pd.concat([results, tempresults])
print(results[['Algorithm','Accuracy']])

#LOGISTIC REGRESSION
lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_pred_lr = lr.predict(X_test)
print(accuracy_score(Y_test, y_pred_lr))
lr_acc = accuracy_score(Y_test, y_pred_lr)

tempresults = pd.DataFrame({'Algorithm': ['Logistic Regression Method'], 'Accuracy': [lr_acc]})
results = pd.concat([results, tempresults])
results[['Algorithm','Accuracy']]

#RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

y_pred_rfc = rfc.predict(X_test)
print(accuracy_score(Y_test, y_pred_rfc))
lr_acc = accuracy_score(Y_test, y_pred_rfc)

tempresults = pd.DataFrame({'Algorithm': ['Random Fores Method'], 'Accuracy': [lr_acc]})
results = pd.concat([results, tempresults])
results[['Algorithm','Accuracy']]

#SUPPORT VECTOR CLASSIFIER
svc = SVC()
svc.fit(X_train, Y_train)

y_pred_svc = svc.predict(X_test)
print(accuracy_score(Y_test, y_pred_svc))
svc_acc = accuracy_score(Y_test, y_pred_svc)

tempresults = pd.DataFrame({'Algorithm': ['Support Vector Classifier'], 'Accuracy': [svc_acc]})
results = pd.concat([results, tempresults])
results[['Algorithm','Accuracy']]

def prediction(fixedacidity=7.4, volatileacidity=0.700, citricacid=0.04, residualsugar=2.0, chlorides=0.075, free_sulfur_dioxide=15, total_sulfur_dioxide=50, density=0.99, pH=3.5, sulphates=0.58, alcohol=9.5):
    temp_array=[]
    temp_array=temp_array+[fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    temp_array=np.array([temp_array])
    return(rfc.predict(temp_array))

final_pred=prediction(fixedacidity=7.8, volatileacidity=0.770, citricacid=0.05, residualsugar=2.5, chlorides=0.077, free_sulfur_dioxide=25, total_sulfur_dioxide=55, density=0.99, pH=4.5, sulphates=0.68, alcohol=9.9)
print(final_pred)
if final_pred>6:
    print("Wine Quality is good")
else:
    print("Wine Quality is not good")