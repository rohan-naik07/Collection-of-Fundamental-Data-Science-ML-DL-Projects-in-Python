# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:53:06 2020

@author: me
"""

import os
import requests
import numpy as np
import tensorflow as tf
from scipy.misc import imsave,imread
from flask import Flask,request,jsonify

with open('fashion_model_flask.json','r') as f:
    model_json=f.read()
    
model=tf.keras.models.model_from_json(model_json)
model.load_weights('fashion_model_flask.h5')

#flask 
app=flask(__name__) # name specifies name of this file
@app.route('/api/v1/<string:img_name>',methods={"POST"})
def classify_image(img_name):
    upload_dir="uploads/"
    image=imread(upload_dir+img_name)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    prediction=model.predict([image.reshape(1,28*28)])
    return jsonify({'object':classes[np.argmax(prediction[0])]})
app.run(port=5000,debug=False)
