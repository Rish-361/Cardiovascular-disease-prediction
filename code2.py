import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import sys


import xgboost as xgb

df = pd.read_csv('Heart.csv')
df = df.dropna(axis=0)
X = df.drop('target', axis=1)
y = df['target']
df.to_csv('data/app_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
new_colspace = ['oldpeak', 'chol', 'age', 'trestbps', 'ca', 'cp', 'thal','sex','exang']
X_train = X_train[new_colspace]
X_test = X_test[new_colspace]
ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=new_colspace)
X_test = pd.DataFrame(ss.transform(X_test), columns=new_colspace)


from flask import Flask,render_template,url_for,request

app = Flask(__name__)
 
loaded_model = pickle.load(open('heart_disease_detector.pkl', 'rb'))
def heart_disease_det(oldpeak,chol,age,trestbps,ca,cp,thal,sex,exang):
    input_data = np.array([oldpeak,chol,age,trestbps,ca,cp,thal,sex,exang])
    input_data.shape=(1,9)
    #input_data = pd.DataFrame(input_data,columns=new_colspace)

    input_data = pd.DataFrame(ss.transform(input_data), columns=new_colspace)
    prediction = loaded_model.predict(input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index3.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        oldpeak = request.form['oldpeak']
        chol = request.form['chol']
        age = request.form['age']
        trestbps = request.form['trestbps']
        ca = request.form['ca']
        cp = request.form['cp']
        thal = request.form['thal']
        sex = request.form['sex']
        exang = request.form['exang']
        pred = heart_disease_det(oldpeak,chol,age,trestbps,ca,cp,thal,sex,exang)
        print(pred)
        
        return render_template('index3.html',prediction=pred)
    else:
        
        return render_template('index3.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
