import os, json
import gzip
import numpy as np

def load_mnist(path, kind='train'):


    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def generate_dataset():

  X_train, y_train = load_mnist('raw_data/fashion', kind='train')
  X_test, y_test = load_mnist('raw_data/fashion', kind='t10k')


  # some simple normalization
  mu = np.mean(X_train.astype(np.float32), 0)
  sigma = np.std(X_train.astype(np.float32), 0)

  X_train = (X_train.astype(np.float32) - mu)/(sigma+0.001)
  X_test = (X_test.astype(np.float32) - mu)/(sigma+0.001)

  return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()


def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_output = "./data/train/mytrain.json"
    test_output = "./data/test/mytest.json"


    X_train, y_train, X_test, y_test = generate_dataset() 

    
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    # label 0: T-shirt(top); 2: pullover; 6: Shirt
    X_train_0 = []
    y_train_0 = []
    X_test_0 = []
    y_test_0 = []

    X_train_2 = []
    y_train_2 = []
    X_test_2 = []
    y_test_2 = []

    X_train_6 = []
    y_train_6 = []
    X_test_6 = []
    y_test_6 = []
    
    for idx, item in enumerate(X_train):
        if y_train[idx] == 0:
            X_train_0.append(X_train[idx])
            y_train_0.append(y_train[idx])
        elif y_train[idx] == 2:
            X_train_2.append(X_train[idx])
            y_train_2.append(y_train[idx]-1)
        elif y_train[idx] == 6:
            X_train_6.append(X_train[idx])
            y_train_6.append(y_train[idx]-4)

    for idx, item in enumerate(X_test):
        if y_test[idx] == 0:
            X_test_0.append(X_test[idx])
            y_test_0.append(y_test[idx])
        elif y_test[idx] == 2:
            X_test_2.append(X_test[idx])
            y_test_2.append(y_test[idx]-1)
        elif y_test[idx] == 6:
            X_test_6.append(X_test[idx])
            y_test_6.append(y_test[idx]-4)


    # for class 0
    train_len = len(X_train_0)
    print("training set for 0: {}".format(train_len))
    test_len = len(X_test_0)
    uname='0:T-shirt(top)'
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X_train_0, 'y': y_train_0}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_test_0, 'y': y_test_0}
    test_data['num_samples'].append(test_len)

    # for class 2

    train_len = len(X_train_2)
    print("training set for 2: {}".format(train_len))
    test_len = len(X_test_2)
    uname='2:pullover'
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X_train_2, 'y': y_train_2}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_test_2, 'y': y_test_2}
    test_data['num_samples'].append(test_len)
    
    # for class 6

    train_len = len(X_train_2)
    print("training set for 6: {}".format(train_len))
    test_len = len(X_test_2)
    uname='6:shirt'
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X_train_6, 'y': y_train_6}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_test_6, 'y': y_test_6}
    test_data['num_samples'].append(test_len)

    with open(train_output,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)
    

if __name__ == "__main__":
    main()

