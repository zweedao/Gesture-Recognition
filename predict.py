import pandas as pd
import numpy as np
import random
import csv
import re
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from joblib import dump, load
from scipy.special import logsumexp
from train import load_imu_data, forward_backward


if __name__ == '__main__':
	## Load data
	gesture_test_folder = "./Test/"
	gesture_model_folder = "./Model/"

	test_files, test_data = load_imu_data(gesture_test_folder)


	## Prediction
	#Use KMeans to cluster IMU data into observation classes
	kmeans = load(gesture_model_folder + 'kmeans.joblib')
	observations_test = {}
	for key, value in test_files.items():
	    observations_test[key] = kmeans.predict(value)

	#Use HMM to predict gesture
	hmm = load(gesture_model_folder + 'hmm.joblib')
	gestures = ['wave','inf','eight','circle','beat3','beat4']
	filenames = [key for key in observations_test.keys()]

	log_likelihoods = np.zeros((len(filenames),len(gestures)))
	predictions = []
	for i, filename in enumerate(filenames):
	    sequence = observations_test[filename]
	    for j, gesture in enumerate(gestures):     
	        log_PI = hmm[gesture]['log_PI']
	        log_A = hmm[gesture]['log_A']
	        log_B = hmm[gesture]['log_B']
	        _, _, log_likelihoods[i,j] = forward_backward(sequence, log_A, log_B, log_PI)
	    
	    #pred_gesture_id = np.argmax(log_likelihoods, axis=1)[i]
	    pred_gesture_ids = np.argsort(-log_likelihoods, axis=1)
	    pred_gestures = [gestures[k] for k in pred_gesture_ids[i,:3]]
	    predictions.append(pred_gestures[0])
	    max_log_likelihood = np.max(log_likelihoods, axis=1)[i]
	    print('File:',filename,'; Prediction:',pred_gestures,'; Log likelihood:',max_log_likelihood)    
	    
	#Accuracy
	correct = [predictions[i] in filenames[i] for i in range(len(filenames))]
	accuracy = np.sum(correct)/len(filenames)
	print('Accuracy:',accuracy)

