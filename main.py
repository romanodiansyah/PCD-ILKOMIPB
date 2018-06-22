from flask import Flask, redirect, request, jsonify,render_template
from PIL import Image
import io
import numpy as np
import os
import cv2
import glob
import csv
import metodePCD as mp
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import shutil
import uuid
import matplotlib.pyplot as plt


app = Flask(__name__)




@app.route('/')
def index():
    return redirect('static/index.html')


@app.route('/data_train', methods=['POST'])
def daun():

	# fixed-sizes for image
	fixed_size = tuple((1000,1000)) #Resize pixel menjadi px x px


	filename = open('data.csv', 'r')
	dataframe = pandas.read_csv(filename)

	kelas = dataframe.drop(dataframe.columns[:-1], axis=1)
	data = dataframe.drop(dataframe.columns[-1:], axis=1)

	# print(data)
	# print(kelas)

	# empty lists to hold feature vectors and labels
	global_features = []
	labels = []

	i, j = 0, 0
	k = 0


	# create all the machine learning models
	models = []
	models.append(('Random Forest',RandomForestClassifier(max_depth=None, random_state=0)))

	# variables to hold the results and names
	results = []
	names = []
	scoring = "accuracy"

	# filter all the warnings
	import warnings
	warnings.filterwarnings('ignore')

	# 10-fold cross validation
	for name, model in models:
	    kfold = KFold(n_splits=10, random_state=7)
	    cv_results = cross_val_score(model, data, kelas, cv=kfold, scoring=scoring)
	    results.append(cv_results)
	    names.append(name)
	    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	    print(msg)


	import matplotlib.pyplot as plt

	# create the model - Random Forests
	clf  =  RandomForestClassifier(max_depth=None, random_state=0)
	# fit the training data to the model
	clf.fit(data,kelas)

	image = cv2.imread('daona2_2.jpg')
	image = cv2.resize(image, fixed_size)
	humoments = mp.hu_moments(image)
	cannywhite = mp.canny(image)
	morphsum = mp.morph(image)
	H,S,V = mp.rataHSV(image)
	diamA, diamB = mp.diameterDetect(image)
	red,green,blue = mp.rataRGB(image)
	global_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB])
	prediction = clf.predict(global_feature.reshape(1,-1))[0]
	return render_template('result.html', prediksi = prediction)

def api_response():
    from flask import jsonify
    if request.method == 'POST':
        return jsonify(**request.json)

@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('daona2_2.jpg')



@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('daona2_2.jpg', 'rb')
    data = fileob.read()
    return data



if __name__ == '__main__':
    app.run(debug=False, port=5010)
