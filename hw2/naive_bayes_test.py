
# coding: utf-8

# In[2]:


import sys
import csv
import numpy as np

#model_file = 'model.csv'
model_file = sys.argv[1] + '.csv'

#test_file = 'spam_data/spam_test.csv'
test_file = sys.argv[2]

#prediction_file = 'spam_data/prediction.csv'
prediction_file = sys.argv[3]

mean_0 = np.ndarray(shape=(56))
std_0 = np.ndarray(shape=(56))
mean_1 = np.ndarray(shape=(56))
std_1 = np.ndarray(shape=(56))

test_X = np.ndarray(shape=(600,56))

#read the model
f = open(model_file, 'r')
i = 0
for row in csv.reader(f):
    j = 0
    for data in row:
        if i == 0:
            mean_0[j] = data
        elif i == 1:
            std_0[j] = data
        elif i == 2:
            mean_1[j] = data
        else:
            std_1[j] = data    
        j += 1
    i += 1
f.close()

#read the testing data
f = open(test_file, 'r')
i = 0
for row in csv.reader(f):
    j = -1
    for data in row:
        if j < 0:
            j += 1
            continue
        if j < 31:
            test_X[i, j] = data
        elif j > 31:
            test_X[i, j - 1] = data
        j += 1
    i += 1
f.close()


# In[3]:

def getClassProb(x, mean, std):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
    prob = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    return np.prod(prob, axis=1)

def predict(x, prob_0, prob_1):
    label = np.ndarray(shape=(x.shape[0], 1))
    for i in range(x.shape[0]):
        if prob_0[i] > prob_1[i]:
            label[i] = 0
        else:
            label[i] = 1
    return label

def getAccu(y, label):
    correct = 0.
    for i in range(y.shape[0]):
        if label[i] == y[i]:
            correct += 1
    return correct / y.shape[0]


# In[4]:

test_prob_0 = getClassProb(test_X, mean_0, std_0)
test_prob_1 = getClassProb(test_X, mean_1, std_1)

test_label = predict(test_X, test_prob_0, test_prob_1)


# In[8]:

label = [['id', 'label']]

for i in range(test_label.shape[0]):
    l = [str(i + 1), str(int(test_label[i]))]
    label.append(l)
    
f = open(prediction_file, 'w')
w = csv.writer(f)
w.writerows(label)
f.close()


# In[ ]:



