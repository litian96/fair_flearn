import scipy.io
import numpy as np
import random
import json
from numpy import *

NUM_USER = 23



# preprocess data (x-mean)/stdev
def preprocess(x):
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    x = (x - means) * 1.0 / std
    where_are_NaNs = isnan(x)
    x[where_are_NaNs] = 0
    return x

def generate_data():
    X = []
    y = []
    mat = scipy.io.loadmat('./raw_data/vehicle.mat')
    raw_x, raw_y = mat['X'], mat['Y']
    print("number of users:", len(raw_x), len(raw_y))
    print("number of features:", len(raw_x[0][0][0]))

    
    for i in range(NUM_USER):
        print("{}-th user has {} samples".format(i, len(raw_x[i][0])))
        #print(len(raw_x[i][0]) * 0.75)
        X.append(preprocess(raw_x[i][0]).tolist())
        y.append(raw_y[i][0].tolist())
        num = 0
        for j in range(len(raw_y[i][0])):
            if raw_y[i][0][j] == 1:
                num += 1
        print("ratio, ", num * 1.0 / len(raw_y[i][0]))
    return X, y
    


def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = "./data/train/mytrain.json"
    test_path = "./data/test/mytest.json"


    X, y = generate_data()

    
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in range(NUM_USER):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples) # the percentage of training samples is 0.75 (in order to be consistant with the statistics shown in the MTL paper)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    

if __name__ == "__main__":
    main()



