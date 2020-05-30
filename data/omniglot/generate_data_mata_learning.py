import os, json
import numpy as np
from PIL import Image



def generate_dataset():

    task_to_class = {}
    for i in range(400): # 400 tasks
        if i < 300: # first 300 meta-training tasks
            class_ids = np.random.choice(1200, 5)
            task_to_class[i]=class_ids
        else: # 
            class_ids = np.random.choice(range(1200, 1623), 5)
            task_to_class[i]=class_ids
    class_to_task={}
    for i in range(1643):
        class_to_task[i] = []
    for i in range(400):
        for j in task_to_class[i]:
            class_to_task[j].append(i)

    X_test = {}  # testing test of all tasks (300 meta-train + 100 meta-test)
    y_test = {}
    X_train = {}  # training set of all tasks (300 meta-train + 100 meta-test)
    y_train = {}

    for i in range(400):
        X_test[i]=[]
        y_test[i]=[]
        X_train[i]=[]
        y_train[i]=[]

    all_data = []
    for idx, f in enumerate(os.listdir("raw/1623_characters")):
        images = os.listdir("raw/1623_characters/"+f+"/")
        for i, img in enumerate(images):
            content = Image.open("raw/1623_characters/"+f+"/"+img, mode='r').convert('L')
            content = content.resize((28, 28))

            #print(np.asarray(content))
            content = np.asarray(content, dtype="int32").flatten()
            all_data.append(content)
            if i < 10:
                for device_id in class_to_task[idx]:
                    X_train[device_id].append(content)
                    y_train[device_id].append(np.where(task_to_class[device_id]==idx)[0][0])

            else:
                for device_id in class_to_task[idx]:
                    X_test[device_id].append(content)
                    y_test[device_id].append(np.where(task_to_class[device_id]==idx)[0][0])


    all_data=np.asarray(all_data)
    print("original data:", all_data[0])
    # some simple normalization
    mu = np.mean(all_data.astype(np.float32), 0)
    print("mu:", mu)
    sigma = np.std(all_data.astype(np.float32), 0)

    for device_id in range(400):
        X_train[device_id] = np.array(X_train[device_id])
        X_test[device_id] = np.array(X_test[device_id])

    for device_id in range(400):
        X_train[device_id] = (X_train[device_id].astype(np.float32) - mu) / (sigma + 0.001)
        X_test[device_id] = (X_test[device_id].astype(np.float32) - mu) / (sigma + 0.001)
        X_train[device_id]=X_train[device_id].tolist()
        X_test[device_id] = X_test[device_id].tolist()

    return X_train, y_train, X_test, y_test


def main():

    train_output = "./data/train/mytrain.json"
    test_output = "./data/test/mytest.json"

    X_train, y_train, X_test, y_test = generate_dataset()
    print("have read in X_train, y_train, X_test, y_test")
    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    num_device = 400 # device is task

    for i in range(num_device):
        uname="class_"+str(i)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}

    with open(train_output, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()


