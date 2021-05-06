# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:45:32 2021

@author: User
"""
import flask
from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

app=Flask(__name__)

filename = 'C:/Users/User/Desktop/new/static/finalized_model.sav'

@app.route('/')
def home():
    return render_template('home.html')
	
@app.route('/about')
def about():
    return render_template('about.html')
	
@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method=='POST':
        
        data=request.form
        int_features = [x for x in data.values()]
        final_features = [np.array(int_features)]
        
        loaded_model = pickle.load(open(filename, 'rb'))
        
        class_labels=['Diabetic', 'Non Diabetic']
        
        result=loaded_model.predict(final_features)
        
    
        return render_template('result.html', data=data, result=result)
         

if __name__ == '__main__':
    app.run(debug=True)