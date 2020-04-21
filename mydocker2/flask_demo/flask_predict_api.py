#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:06:35 2018

@author: vivekkalyanarangan
"""

print('Namespace started')

import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd
import os


print('Namespace loaded')


for subdir, dirs, files in os.walk('./'):
    for file in files:
      print(file)

path = '/var/www/flask_predict_api/rf.pkl'
path = '/home/user/Samples/mydocker2/rf.pkl'
path = 'rf.pkl'
print('pickle started')
with open(path, 'rb') as model_file:
    model = pickle.load(model_file)
print('pickle completed')
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    
    print('test')
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    print(prediction)
    return str(prediction[0])

@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))

@app.route("/")
def main():
    return "Python Hello on " + os.getenv('HOSTNAME', "unknown") + "\n"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6005, debug=True)
    #app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    