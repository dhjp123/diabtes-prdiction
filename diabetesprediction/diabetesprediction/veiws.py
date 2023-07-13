from django.shortcuts import render
import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    diabetes_df = pd.read_csv(r"D:\dsa_new\diabetes (1).csv")
    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                        random_state=1)
    model = RandomForestClassifier(n_estimators=300, bootstrap=True, max_features='sqrt')
    model.fit(X_train, y_train)


    val1=  float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""

    if pred == [1]:
        result1 = "This Person have Diabetic"
    else:
        result1 = "This person have No Diabetic"
    return render(request, 'predict.html', {"result2": result1})