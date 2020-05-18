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


def load_imu_data(gesture_folder):
    files = {}
    data = np.empty([0,6])
    for file in os.listdir(gesture_folder):
        if file[0] != '.':
            print(file)
            filename = file.split('.')[0]
            imu_data = np.loadtxt(gesture_folder + file)    
            files[filename] = imu_data[:,1:]
            data = np.append(data, imu_data[:,1:], axis=0)
        
    return files, data


def forward_backward(sequence, log_A, log_B, log_PI):
        observed_classes, hidden_states = log_B.shape
        times = sequence.shape[0]
        
        #Forward
        log_alpha = np.zeros((times, hidden_states))
        log_alpha[0] = log_PI + log_B[sequence[0]]
        for t in range (times - 1):
            log_alpha[t+1] = logsumexp(log_alpha[t].reshape((1,-1)) + log_A, axis=1) + log_B[sequence[t+1]]            
        
        log_likelihood = logsumexp(log_alpha[times-1])
        
        #Backward
        log_beta = np.zeros((times, hidden_states))
        log_beta[times-1] = np.log(1)
        for t in range(times-2, -1, -1):
            log_beta[t] = logsumexp(log_beta[t+1].reshape((-1,1)) + log_A + log_B[sequence[t+1]].reshape((-1,1)), axis=0)
        
        return log_alpha, log_beta, log_likelihood


if __name__ == '__main__':
    ## Load data
    gesture_train_folder = "./Train/"
    gesture_model_folder = "./Model/"
    train_files, train_data = load_imu_data(gesture_train_folder)

    ## Train Hidden Markove Model
    hmm_models = {}

    #Hyper parameters
    gestures = ['wave','inf','eight','circle','beat3','beat4']
    hidden_states = 14 
    observed_classes = 20
    epochs = 30
    threshold = 0.1

    #Use KMeans to cluster IMU data into observation classes
    kmeans = KMeans(n_clusters = observed_classes)
    kmeans.fit(train_data)
    clusters = kmeans.labels_
    dump(kmeans, gesture_model_folder + 'kmeans.joblib')

    observations = {}
    for key, value in train_files.items():
        observations[key] = kmeans.predict(value)

    #Initial State Distribution (pi)
    state_prob = np.ones(hidden_states)/hidden_states 

    #Transition probabilities (A)
    transit_prob = np.random.uniform(low=0.1, high=1, size=(hidden_states, hidden_states))
    transit_prob = np.tril(transit_prob)
    transit_prob /= np.sum(transit_prob, axis=0)

    #Observation probabilities (B)
    observed_prob = np.random.uniform(low=0.1, high=1, size=(observed_classes, hidden_states))
    observed_prob /= np.sum(observed_prob, axis=0)

    #HMM model
    for gesture in gestures:
        #gesture = gestures[0] #test
        log_PI = np.log(state_prob)
        log_A = np.log(transit_prob)
        log_B = np.log(observed_prob)
        observed_sequences = [observations[key] for key in observations.keys() if gesture in key]
        
        last_log_likelihood = None
        for i in range(epochs):
            #i=0 #test
            first_gamma = []
            all_gamma = []
            all_xi = []
            all_obs_states = []
            log_likelihood = np.zeros(len(observed_sequences))
        
            for j, sequence in enumerate(observed_sequences):
                #j, sequence = 0, observed_sequences[0] #test            
                times = sequence.shape[0]
                
                #Forward - Backward
                log_alpha, log_beta, log_likelihood[j] = forward_backward(sequence, log_A, log_B, log_PI)
                
                #E-step
                log_gamma = (log_alpha + log_beta).transpose()
                log_gamma -= logsumexp(log_gamma, axis=0).reshape((1,-1))
                
                log_xi = np.zeros((hidden_states, hidden_states, times-1))
                for t in range(times-1):
                    log_xi[:, :, t] = log_alpha[t].reshape((1,-1)) + log_A + log_B[sequence[t+1]].reshape((-1,1)) + log_beta[t+1].reshape((-1,1))
                log_xi -= logsumexp(log_xi, axis=2).reshape((hidden_states,hidden_states,1))
                log_xi[np.isnan(log_xi)] = -np.inf 
                
                first_gamma.append(log_gamma[:,0])
                all_gamma.append(logsumexp(log_gamma, axis=1))
                all_xi.append(logsumexp(log_xi, axis=2))
                
                obs_states = np.zeros((observed_classes, hidden_states))
                for c in range(observed_classes):
                    try:
                        obs_states[c] = logsumexp(log_gamma[:, sequence == c], axis=1)
                    except ValueError: # no observation c in sequence
                        obs_states[c] = np.full(hidden_states, -np.inf)
                all_obs_states.append(obs_states)
                
            avg_log_likelihood = np.average(log_likelihood)
            print('Gesture:', gesture, ';Epoch:', i+1, ';Log likelihood:', avg_log_likelihood)                
            
            first_gamma_sum = logsumexp(np.array(first_gamma), axis=0)
            gamma_sum = logsumexp(np.array(all_gamma), axis=0).reshape((1,-1))
            xi_sum = logsumexp(np.array(all_xi), axis=0)
            obs_states_sum = logsumexp(np.array(all_obs_states), axis=0)

            ## M-step
            log_A = xi_sum - gamma_sum
            log_A -= logsumexp(log_A, axis=0)
            
            log_B = obs_states_sum - gamma_sum
            log_B -= logsumexp(log_B, axis=0)
            
            log_PI = first_gamma_sum
            log_PI -= logsumexp(log_PI)
                   
            #Stop condition
            if last_log_likelihood is not None and abs(avg_log_likelihood - last_log_likelihood) < threshold:
                break
            last_log_likelihood = np.average(log_likelihood)

        #save into HMM model
        hmm_models[gesture] = {'log_PI': log_PI, 'log_A': log_A, 'log_B': log_B}

    #export HMM model
    dump(hmm_models, gesture_model_folder + 'hmm.joblib')
